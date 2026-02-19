"""
mlp.py
──────
Small MLP that maps log‖∇E(q)‖ → step-size multiplier α.

Design decisions documented inline.
"""

import jax
import jax.numpy as jnp
import jax.random as jr


def init_mlp(key, layer_sizes=(1, 16, 16, 1)):
    """
    He-initialised MLP stored as a list of {"W", "b"} dicts.

    JAX treats list-of-dicts as a pytree automatically, so these params can
    live inside scan carries and be differentiated through without any
    special registration.

    Output bias initialised to softplus⁻¹(1) ≈ 0.541 so the network
    starts predicting α ≈ 1 (no-op) before training begins.  This ensures
    the very first warm-up steps are equivalent to standard HMC, giving a
    safe starting point before the MLP has learned anything.

    Weights are scaled by 0.1 × He-init to keep the network in the linear
    regime of tanh initially, which speeds up early learning.
    """
    params   = []
    keys     = jr.split(key, len(layer_sizes) - 1)
    n_layers = len(layer_sizes) - 1

    for i, (k, n_in, n_out) in enumerate(
        zip(keys, layer_sizes[:-1], layer_sizes[1:])
    ):
        W = jr.normal(k, (n_in, n_out)) * (jnp.sqrt(2.0 / n_in) * 0.1)
        b = jnp.ones(n_out) * 0.541 if i == n_layers - 1 else jnp.zeros(n_out)
        params.append({"W": W, "b": b})

    return params


def mlp_forward(params, log_grad_norm):
    """
    Forward pass: scalar log_grad_norm → positive scalar α.

    Architecture: tanh hidden layers + softplus output.
    Softplus is used (not ReLU/sigmoid) because:
      - It is smooth everywhere → clean gradients during training
      - Its range is (0, ∞)    → α is always positive
      - softplus(x) ≈ x for large x → no saturation for large inputs

    Clipping to [0.10, 2.50] provides hard safety bounds:
      - Lower bound 0.10: prevents α collapsing to ~0, which would stall
        the chain completely (0% acceptance is caused by q'≈q, not rejection)
      - Upper bound 2.50: with base_step_size=0.2, ε_eff ≤ 0.5, which keeps
        the 10-step leapfrog within the integrator's stability region on
        Rosenbrock (stability requires ε√λ_max < 2; λ_max ≈ 8 → ε < 0.71)

    ⚠ Known limitation (Q1 in theoretical review):
      Using α computed at q₀ for the entire trajectory breaks time-reversal
      symmetry.  The MH acceptance ratio exp(-ΔH) is then biased.  See
      leapfrog.py for the symmetric-step-size fix (Fix B).
    """
    h = log_grad_norm.reshape(1)

    for layer in params[:-1]:
        h = jnp.tanh(h @ layer["W"] + layer["b"])

    raw   = (h @ params[-1]["W"] + params[-1]["b"]).squeeze()
    alpha = jax.nn.softplus(raw)
    return jnp.clip(alpha, 0.10, 2.50)


def adam_update(params, grads, m, v, t, lr,
                beta1=0.9, beta2=0.999, eps=1e-8,
                max_grad_norm=1.0):
    """
    Single Adam step on an arbitrary JAX pytree.

    Includes two numerical safety measures:

    FIX 3 — Global gradient clipping:
      Clips the entire gradient pytree so its ℓ₂ norm ≤ max_grad_norm.
      Without this, one diverged leapfrog trajectory sends grads → ∞ and
      permanently corrupts the Adam moment buffers.

    FIX 4 — NaN guard on updated params:
      After the parameter update, any leaf that became nan/inf is replaced
      with the previous value.  This is a last-resort safety net; FIX 3
      should prevent it from ever triggering.
    """
    # Global gradient clipping
    leaves      = jax.tree_util.tree_leaves(grads)
    global_norm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in leaves))
    scale       = jnp.minimum(1.0, max_grad_norm / (global_norm + 1e-8))
    grads       = jax.tree_util.tree_map(lambda g: g * scale, grads)

    t_new = t + 1.0

    m_new = jax.tree_util.tree_map(
        lambda mi, gi: beta1 * mi + (1.0 - beta1) * gi, m, grads)
    v_new = jax.tree_util.tree_map(
        lambda vi, gi: beta2 * vi + (1.0 - beta2) * gi ** 2, v, grads)

    m_hat = jax.tree_util.tree_map(
        lambda mi: mi / (1.0 - beta1 ** t_new), m_new)
    v_hat = jax.tree_util.tree_map(
        lambda vi: vi / (1.0 - beta2 ** t_new), v_new)

    params_candidate = jax.tree_util.tree_map(
        lambda pi, mhi, vhi: pi - lr * mhi / (jnp.sqrt(vhi) + eps),
        params, m_hat, v_hat)

    # NaN guard: revert any leaf that went non-finite
    params_safe = jax.tree_util.tree_map(
        lambda new, old: jnp.where(jnp.all(jnp.isfinite(new)), new, old),
        params_candidate, params)

    return params_safe, m_new, v_new, t_new
