"""
distributions.py
────────────────
Benchmark log-probability functions used throughout the ALCS-HMC project.

Both functions are unnormalised: they return log p(θ) up to an additive
constant.  JAX can differentiate through them automatically.
"""

import jax.numpy as jnp


def log_prob_rosenbrock(theta):
    """
    Rosenbrock (banana) distribution.

        log p(x, y) ∝ −(1−x)²/20 − (y − x²)²

    The high-probability region is a thin, curved ridge.  This tests how
    well a sampler can follow strong, non-linear correlations.

    True marginal moments (analytically):
        E[x]   = 1,       Var[x]  = 10   →  sd[x]  ≈ 3.16
        E[y]   = 11,      Var[y]  ≈ 42   →  sd[y]  ≈ 6.5
    """
    x, y = theta[0], theta[1]
    return -0.05 * (1.0 - x) ** 2 - (y - x ** 2) ** 2


def log_prob_funnel(theta):
    """
    Neal's Funnel distribution.

        v  ~ N(0, 9)
        x  ~ N(0, exp(v))

    The funnel neck (v << 0) requires tiny step sizes; the mouth (v >> 0)
    requires large ones.  A single fixed step size cannot handle both.

    True marginal moments:
        E[v]  = 0,   sd[v] = 3
        E[x]  = 0,   sd[x] = sqrt(E[exp(v)]) = sqrt(exp(9/2)) ≈ 9.49
    """
    v, x = theta[0], theta[1]
    log_p_v       = -0.5 * v ** 2 / 9.0
    log_p_x_giv_v = -0.5 * x ** 2 * jnp.exp(-v) - 0.5 * v
    return log_p_v + log_p_x_giv_v


# Ground-truth moments for scoring samplers against
ROSENBROCK_TRUE = {"x": {"mean": 1.0,  "sd": 3.162},
                   "y": {"mean": 11.0, "sd": 6.5}}

FUNNEL_TRUE     = {"v": {"mean": 0.0,  "sd": 3.0},
                   "x": {"mean": 0.0,  "sd": 9.487}}
