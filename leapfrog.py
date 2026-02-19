"""
leapfrog.py
───────────
Leapfrog (velocity-Verlet) integrators and associated loss functions.

Two variants are provided:
    leapfrog        — static n_steps (lax.scan),  AD-compatible
    leapfrog_symm   — static n_steps + symmetric step size (Fix B)

The symmetric variant addresses the time-reversal flaw identified in the
theoretical review: when ε depends on position, using ε(q₀) for the full
trajectory breaks the reversibility of the proposal.  leapfrog_symm
averages ε(q₀) and ε(qL) to restore symmetry at the cost of one extra
gradient evaluation after the trajectory.

Loss functions:
    energy_error_loss  — original (|ΔH| − ΔH_target)²
    esjd_aware_loss    — ESJD surrogate + acceptance regularisation
                         Avoids the α→0 degeneracy of the energy-error loss.
"""

import jax
import jax.numpy as jnp

from mlp import mlp_forward


# ── Standard leapfrog (AD-compatible, static n_steps) ────────────────────────

def leapfrog(q, p, log_prob_fn, step_size, n_steps):
    """
    Velocity-Verlet integrator.  n_steps must be a static Python int so
    that lax.scan can set a compile-time loop length.  This is required
    when the function is differentiated via jax.value_and_grad.

    NaN safety (FIX 2): each scan step freezes (q, p) if either becomes
    non-finite, preventing NaN propagation through the rest of the trajectory
    while keeping ΔH finite so the MH test can reject cleanly.

    ⚠  Time-reversal note: step_size is held constant through the trajectory.
       If step_size = ε_base × α(q₀), the proposal is asymmetric and the
       MH acceptance ratio exp(-ΔH) is biased.  Use leapfrog_symm for
       a corrected version.
    """
    grad_fn = jax.grad(log_prob_fn)

    p = p + 0.5 * step_size * grad_fn(q)

    def body_fn(carry, _):
        q, p = carry
        q_new = q + step_size * p
        g     = grad_fn(q_new)
        p_new = p + step_size * g

        both_finite = (jnp.all(jnp.isfinite(q_new)) &
                       jnp.all(jnp.isfinite(p_new)))
        q_out = jnp.where(both_finite, q_new, q)
        p_out = jnp.where(both_finite, p_new, p)
        return (q_out, p_out), None

    (q, p), _ = jax.lax.scan(body_fn, (q, p), None, length=n_steps - 1)

    q = q + step_size * p
    p = p + 0.5 * step_size * grad_fn(q)
    return q, p


# ── Symmetric step-size leapfrog (Fix B — addresses reversibility) ────────────

def leapfrog_symm(q, p, log_prob_fn, mlp_params, base_step_size,
                  n_steps, log_grad_norm_start):
    """
    Symmetric-step leapfrog that restores time-reversal symmetry.

    Procedure:
      1. Run the trajectory with ε(q₀) to obtain the proposal qL.
      2. Compute ε(qL) at the end-point.
      3. Set ε_sym = (ε(q₀) + ε(qL)) / 2.
      4. Re-run the trajectory with ε_sym.

    The re-run makes the proposal symmetric:
        forward:  q₀ → qL  with ε_sym(q₀, qL)
        reverse:  qL → q₀  with ε_sym(qL, q₀) = ε_sym(q₀, qL)  ✓

    This costs one extra gradient evaluation (to get ∇E(qL)) and one
    extra leapfrog run (n_steps extra gradient evaluations).  Total
    overhead: (n_steps + 1) extra gradient calls per step.

    Note: this function is NOT used in the training loss — the training
    loss uses the plain leapfrog to keep second-order AD tractable.
    """
    grad_fn = jax.grad(log_prob_fn)

    # Step 1: pilot trajectory with ε(q₀)
    alpha_start = mlp_forward(mlp_params, log_grad_norm_start)
    eps_start   = base_step_size * alpha_start
    q_pilot, _  = leapfrog(q, p, log_prob_fn, eps_start, n_steps)

    # Step 2: ε at proposal end-point
    grad_qL         = grad_fn(q_pilot)
    log_gnorm_end   = jnp.log(jnp.linalg.norm(grad_qL) + 1e-8)
    alpha_end       = mlp_forward(mlp_params, log_gnorm_end)
    eps_end         = base_step_size * alpha_end

    # Step 3: symmetric step size
    eps_sym = 0.5 * (eps_start + eps_end)

    # Step 4: final trajectory with symmetric ε
    q_prop, p_prop = leapfrog(q, p, log_prob_fn, eps_sym, n_steps)
    return q_prop, p_prop, eps_sym


# ── Loss functions ────────────────────────────────────────────────────────────

def energy_error_loss(mlp_params, q, p, log_prob_fn,
                      base_step_size, n_leapfrog,
                      log_grad_norm, target_delta_H):
    """
    Original energy-error loss:  L(θ) = ( |ΔH| − ΔH_target )²

    Training signal:
        |ΔH| > target → ε_eff too large → gradient shrinks α
        |ΔH| < target → ε_eff too small → gradient grows  α

    ⚠  Known degeneracy (Q2 in theoretical review):
       α → 0 satisfies the loss perfectly (|ΔH| → 0) but stalls the chain.
       The α clip lower bound (0.10) prevents complete collapse but does
       not remove the attractive basin near the lower bound.
       Use esjd_aware_loss to avoid this degeneracy.
    """
    alpha    = mlp_forward(mlp_params, log_grad_norm)
    eff_step = base_step_size * alpha
    q_prop, p_prop = leapfrog(q, p, log_prob_fn, eff_step, n_leapfrog)

    H_curr = -log_prob_fn(q)      + 0.5 * jnp.dot(p, p)
    H_prop = -log_prob_fn(q_prop) + 0.5 * jnp.dot(p_prop, p_prop)
    delta_H = H_prop - H_curr

    delta_H_safe = jnp.where(jnp.isfinite(delta_H), delta_H, 50.0)
    return (jnp.abs(delta_H_safe) - target_delta_H) ** 2


def esjd_aware_loss(mlp_params, q, p, log_prob_fn,
                    base_step_size, n_leapfrog,
                    log_grad_norm, target_acceptance):
    """
    ESJD-aware loss that avoids the α→0 degeneracy.

        L(θ) = −ESJD_surrogate + λ · (accept − target_acceptance)²

    where:
        ESJD_surrogate = soft_accept × ‖q' − q‖²
        soft_accept    = min(1, exp(−ΔH))   (differentiable MH surrogate)

    The −ESJD term rewards large, accepted moves.
    The acceptance penalty keeps the acceptance rate near target_acceptance,
    preventing both α→0 (no movement) and α→∞ (all rejected).

    The α→0 degeneracy is broken because ESJD → 0 as ‖q'−q‖ → 0,
    so the gradient always pushes α upward from the lower clip boundary.
    """
    alpha    = mlp_forward(mlp_params, log_grad_norm)
    eff_step = base_step_size * alpha
    q_prop, p_prop = leapfrog(q, p, log_prob_fn, eff_step, n_leapfrog)

    H_curr  = -log_prob_fn(q)      + 0.5 * jnp.dot(p, p)
    H_prop  = -log_prob_fn(q_prop) + 0.5 * jnp.dot(p_prop, p_prop)
    delta_H = jnp.where(
        jnp.isfinite(H_prop - H_curr), H_prop - H_curr, 50.0)

    # Differentiable acceptance surrogate (clamp avoids exp overflow)
    soft_accept = jnp.minimum(
        1.0, jnp.exp(jnp.clip(-delta_H, -50.0, 0.0)))

    sq_jump     = jnp.sum((q_prop - q) ** 2)
    esjd        = soft_accept * sq_jump
    acc_penalty = (soft_accept - target_acceptance) ** 2

    return -esjd + 0.5 * acc_penalty


# Pre-bound value_and_grad for both loss functions
loss_and_grad_energy = jax.value_and_grad(energy_error_loss,  argnums=0)
loss_and_grad_esjd   = jax.value_and_grad(esjd_aware_loss,    argnums=0)
