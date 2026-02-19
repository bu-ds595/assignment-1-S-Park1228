"""
sampler.py
──────────
ALCS-HMC v1: the numerically stable single-scalar-input sampler.

This is the version confirmed to work (79.95% acceptance on Rosenbrock).
It exposes two corrective flags motivated by the theoretical review:

    use_symm_step : bool  — use leapfrog_symm to restore time-reversal
                            symmetry (Fix B, addresses Q1)
    loss_type     : str   — "energy_error" (original) or "esjd" (addresses Q2)

The ablation runner (ablation.py) sweeps over these flags alongside the
original hyperparameters to isolate what each design decision contributes.
"""

import jax
import jax.numpy as jnp
import jax.random as jr

from mlp      import init_mlp, mlp_forward, adam_update
from leapfrog import (leapfrog, leapfrog_symm,
                      loss_and_grad_energy, loss_and_grad_esjd)


def run_alcs_hmc(
    key,
    log_prob_fn,
    initial_position,
    base_step_size    = 0.2,
    n_leapfrog        = 10,
    n_samples         = 50_000,
    n_warmup          = None,
    lr_mlp            = 3e-4,
    target_acceptance = 0.75,
    mlp_layer_sizes   = (1, 16, 16, 1),
    loss_type         = "energy_error",   # "energy_error" | "esjd"
    use_symm_step     = False,            # True → Fix B (symmetric ε)
):
    """
    ALCS-HMC v1 — two-phase adaptive sampler.

    Phase 1 — Warm-up (steps 0 … n_warmup−1):
        MLP is updated via one Adam step per iteration using the chosen loss.

    Phase 2 — Sampling (steps n_warmup … n_samples−1):
        MLP frozen.  α(log‖∇E(q)‖) is applied as a static lookup.

    Returns
    ───────
    dict with keys:
        samples              (n_samples, D)
        accept_rate_warmup   scalar
        accept_rate_sampling scalar   ← main performance metric
        alphas               (n_samples,)
        warmup_losses        (n_warmup,)
        mlp_params           final MLP weights
    """
    if n_warmup is None:
        n_warmup = n_samples // 5

    D              = initial_position.shape[0]
    target_delta_H = -jnp.log(jnp.array(target_acceptance))

    # Select loss function (captured by closure — no overhead in scan)
    if loss_type == "esjd":
        _loss_and_grad = loss_and_grad_esjd
        def _loss_call(mlp_p, q, p, log_gnorm, inv_target):
            # esjd_aware_loss takes target_acceptance, not target_delta_H
            return _loss_and_grad(mlp_p, q, p, log_prob_fn,
                                  base_step_size, n_leapfrog,
                                  log_gnorm, target_acceptance)
    else:
        _loss_and_grad = loss_and_grad_energy
        def _loss_call(mlp_p, q, p, log_gnorm, inv_target):
            return _loss_and_grad(mlp_p, q, p, log_prob_fn,
                                  base_step_size, n_leapfrog,
                                  log_gnorm, target_delta_H)

    key, key_mlp = jr.split(key)
    mlp_params   = init_mlp(key_mlp, mlp_layer_sizes)
    adam_m = jax.tree_util.tree_map(jnp.zeros_like, mlp_params)
    adam_v = jax.tree_util.tree_map(jnp.zeros_like, mlp_params)
    adam_t = jnp.array(0.0, dtype=jnp.float32)

    def one_step(carry, key):
        q, mlp_p, m, v, t, step_idx = carry
        key_p, key_mh = jr.split(key)

        grad_q        = jax.grad(log_prob_fn)(q)
        log_grad_norm = jnp.log(jnp.linalg.norm(grad_q) + 1e-8)

        alpha = mlp_forward(mlp_p, log_grad_norm)
        p     = jr.normal(key_p, shape=(D,))

        # Proposal — symmetric or standard
        if use_symm_step:
            q_prop, p_prop, eff_step = leapfrog_symm(
                q, p, log_prob_fn, mlp_p, base_step_size,
                n_leapfrog, log_grad_norm)
        else:
            eff_step       = base_step_size * alpha
            q_prop, p_prop = leapfrog(q, p, log_prob_fn, eff_step, n_leapfrog)

        # MH accept / reject
        H_curr  = -log_prob_fn(q)      + 0.5 * jnp.dot(p, p)
        H_prop  = -log_prob_fn(q_prop) + 0.5 * jnp.dot(p_prop, p_prop)
        delta_H = H_prop - H_curr

        accept_prob = jnp.minimum(
            1.0, jnp.exp(jnp.clip(-delta_H, -50.0, 50.0)))
        proposal_ok = (jnp.all(jnp.isfinite(q_prop)) &
                       jnp.isfinite(accept_prob))
        is_accepted = (jr.uniform(key_mh) < accept_prob) & proposal_ok
        q_new       = jnp.where(is_accepted, q_prop, q)

        # MLP warm-up update
        def warmup_update(_):
            loss_val, grads = _loss_call(
                mlp_p,
                jax.lax.stop_gradient(q),
                jax.lax.stop_gradient(p),
                jax.lax.stop_gradient(log_grad_norm),
                None,
            )
            new_p, new_m, new_v, new_t = adam_update(
                mlp_p, grads, m, v, t, lr_mlp)
            return new_p, new_m, new_v, new_t, loss_val

        def frozen_pass(_):
            loss_val = _loss_call(
                mlp_p,
                jax.lax.stop_gradient(q),
                jax.lax.stop_gradient(p),
                jax.lax.stop_gradient(log_grad_norm),
                None,
            )[0]
            return mlp_p, m, v, t, loss_val

        mlp_p_new, m_new, v_new, t_new, loss_val = jax.lax.cond(
            step_idx < n_warmup, warmup_update, frozen_pass, operand=None)

        carry_new = (q_new, mlp_p_new, m_new, v_new, t_new, step_idx + 1)
        outputs   = (q_new, accept_prob, alpha, loss_val)
        return carry_new, outputs

    keys = jr.split(key, n_samples)
    initial_carry = (
        initial_position, mlp_params,
        adam_m, adam_v, adam_t,
        jnp.array(0, dtype=jnp.int32),
    )

    final_carry, (samples, accept_probs, alphas, losses) = jax.lax.scan(
        one_step, initial_carry, keys)

    return dict(
        samples              = samples,
        accept_rate_warmup   = float(accept_probs[:n_warmup].mean()),
        accept_rate_sampling = float(accept_probs[n_warmup:].mean()),
        alphas               = alphas,
        warmup_losses        = losses[:n_warmup],
        mlp_params           = final_carry[1],
        n_warmup             = n_warmup,
    )
