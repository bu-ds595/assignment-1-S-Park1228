# =============================================================================
# ALCS-HMC: Adaptive Local-Curvature Step-size HMC
# =============================================================================
# Drop this cell into the starter notebook after the HMC baseline.
#
# Design overview
# ───────────────
#   1. Neural Controller  — a tiny MLP that maps  log‖∇E(q)‖  →  α > 0
#   2. Adaptive Leapfrog  — uses  ε_eff = ε_base · α  at every sub-step
#   3. MH correction      — keeps the chain asymptotically correct even
#                           though variable ε breaks symplecticity
#   4. Warm-up training   — maximises E[Squared Jump Distance] via Adam
#                           while the chain explores, then freezes weights
# =============================================================================

import jax
import jax.numpy as jnp
import jax.random as jr
import optax


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Neural Controller
# ─────────────────────────────────────────────────────────────────────────────

def init_mlp_params(key, hidden_dim: int = 8) -> dict:
    """
    Initialise weights for a 1 → hidden_dim → 1 MLP.

    Architecture:
        input  : log‖∇E(q)‖  (1 scalar)
        hidden : tanh units
        output : softplus(·) + ε  →  α ∈ (0, ∞)
    """
    k1, k2 = jr.split(key)
    return {
        "W1": jr.normal(k1, (1, hidden_dim)) * 0.1,   # (1, H)
        "b1": jnp.zeros(hidden_dim),                   # (H,)
        "W2": jr.normal(k2, (hidden_dim, 1)) * 0.1,   # (H, 1)
        "b2": jnp.zeros(1),                            # (1,)
    }


def mlp_forward(params: dict, log_grad_norm: float) -> float:
    """
    Forward pass of the neural controller.

    Args:
        params:        MLP weight dict (W1, b1, W2, b2)
        log_grad_norm: log‖∇E(q)‖  (scalar)

    Returns:
        alpha: step-size multiplier > 0
    """
    x   = jnp.array([log_grad_norm])                    # (1,)
    h   = jnp.tanh(x @ params["W1"] + params["b1"])     # (H,)
    out = (h @ params["W2"] + params["b2"])[0]          # scalar
    return jax.nn.softplus(out) + 1e-3                   # α > 0


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Adaptive Leapfrog Integrator
# ─────────────────────────────────────────────────────────────────────────────

def leapfrog_adaptive(q, p, log_prob_fn, mlp_params, epsilon_base, n_steps):
    """
    L-step leapfrog with a neural-network–adapted step size.

    At each sub-step the effective step size is
        ε_eff(q) = ε_base · α(log‖∇log_p(q)‖)
    where α is produced by the MLP controller.

    Note: Variable ε breaks exact symplecticity, so the Hamiltonian is
    no longer exactly conserved.  The MH correction in `alcs_hmc_kernel`
    accounts for this — the chain is still asymptotically exact.

    Args:
        q, p:        current position / momentum  (D,)
        log_prob_fn: scalar log-prob function
        mlp_params:  neural controller weights
        epsilon_base: base step size ε_base
        n_steps:     number of leapfrog steps L

    Returns:
        q_prop, p_prop: proposed position / momentum
    """
    grad_fn = jax.grad(log_prob_fn)

    def get_grad_and_eps(pos):
        g       = grad_fn(pos)
        log_gn  = jnp.log(jnp.linalg.norm(g) + 1e-8)
        alpha   = mlp_forward(mlp_params, log_gn)
        return g, epsilon_base * alpha

    # ── initial half-step for momentum ───────────────────────────────────
    g0, eps0 = get_grad_and_eps(q)
    p = p + 0.5 * eps0 * g0

    # ── L-1 full leapfrog steps via lax.scan ─────────────────────────────
    def step(carry, _):
        q, p = carry
        g,  eps  = get_grad_and_eps(q)
        q        = q + eps * p
        g2, eps2 = get_grad_and_eps(q)
        p        = p + eps2 * g2
        return (q, p), None

    (q, p), _ = jax.lax.scan(step, (q, p), None, length=n_steps - 1)

    # ── final full position step ──────────────────────────────────────────
    g_f, eps_f = get_grad_and_eps(q)
    q = q + eps_f * p

    # ── final half-step for momentum ──────────────────────────────────────
    g_end, eps_end = get_grad_and_eps(q)
    p = p + 0.5 * eps_end * g_end

    return q, p


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Single ALCS-HMC Transition (with MH correction)
# ─────────────────────────────────────────────────────────────────────────────

def alcs_hmc_kernel(key, q, log_prob_fn, mlp_params, epsilon_base, n_steps):
    """
    One ALCS-HMC Metropolis-Hastings step.

    1. Draw momentum  p ~ N(0, I)
    2. Propose (q', p') via adaptive leapfrog
    3. Accept / reject via standard HMC MH ratio

    Returns:
        q_new:       next position
        accepted:    bool
        accept_prob: float ∈ [0, 1]
    """
    key_p, key_mh = jr.split(key)
    D = q.shape[0]

    p        = jr.normal(key_p, (D,))
    H_curr   = -log_prob_fn(q) + 0.5 * jnp.dot(p, p)

    q_prop, p_prop = leapfrog_adaptive(
        q, p, log_prob_fn, mlp_params, epsilon_base, n_steps
    )
    H_prop   = -log_prob_fn(q_prop) + 0.5 * jnp.dot(p_prop, p_prop)

    log_acc  = jnp.minimum(0.0, H_curr - H_prop)
    acc_prob = jnp.exp(log_acc)
    accepted = jr.uniform(key_mh) < acc_prob

    q_new    = jnp.where(accepted, q_prop, q)
    return q_new, accepted, acc_prob


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Warm-up: Train the Neural Controller via ESJD Maximisation
# ─────────────────────────────────────────────────────────────────────────────

def warmup_train_controller(
    key,
    log_prob_fn,
    initial_position,
    epsilon_base,
    n_leapfrog,
    n_warmup:   int   = 500,
    lr:         float = 3e-3,
    hidden_dim: int   = 8,
    print_every: int  = 100,
):
    """
    Train the MLP controller during a warm-up phase.

    Objective: maximise Expected Squared Jump Distance (ESJD)
        ESJD(params) = E_p[ ‖q' - q‖² · min(1, exp(H_curr − H_prop)) ]

    Since q' and accept_prob are both differentiable functions of mlp_params
    (through the leapfrog), we can backprop directly through this expression.

    Strategy: at each warm-up step
        (a) compute ∇_params (-ESJD)  at current chain position q
        (b) update params via Adam
        (c) advance the chain by one ALCS-HMC step

    Args:
        key:              JAX random key
        log_prob_fn:      log p(θ)
        initial_position: starting position  (D,)
        epsilon_base:     base leapfrog step size
        n_leapfrog:       leapfrog steps per proposal
        n_warmup:         number of warm-up gradient steps
        lr:               Adam learning rate
        hidden_dim:       MLP hidden width
        print_every:      log interval (set to 0 to silence)

    Returns:
        mlp_params:      trained parameter dict  (frozen after this)
        warmup_samples:  array  (n_warmup, D)   for diagnostics
    """
    key_init, key_run = jr.split(key)
    mlp_params = init_mlp_params(key_init, hidden_dim)
    optimizer  = optax.adam(lr)
    opt_state  = optimizer.init(mlp_params)

    # ── differentiable negative ESJD loss ────────────────────────────────
    def neg_esjd(params, q, key_loss):
        p        = jr.normal(key_loss, q.shape)
        H_curr   = -log_prob_fn(q) + 0.5 * jnp.dot(p, p)
        q_prop, p_prop = leapfrog_adaptive(
            q, p, log_prob_fn, params, epsilon_base, n_leapfrog
        )
        H_prop   = -log_prob_fn(q_prop) + 0.5 * jnp.dot(p_prop, p_prop)
        acc_prob = jnp.exp(jnp.minimum(0.0, H_curr - H_prop))
        esjd     = jnp.sum((q_prop - q) ** 2) * acc_prob
        return -esjd  # minimise → maximise ESJD

    # JIT-compile the loss+grad for speed
    loss_and_grad = jax.jit(jax.value_and_grad(neg_esjd))

    q = initial_position
    warmup_samples = []

    for i, k in enumerate(jr.split(key_run, n_warmup)):
        k_loss, k_step = jr.split(k)

        # (a) gradient step on MLP params
        loss_val, grads = loss_and_grad(mlp_params, q, k_loss)
        updates, opt_state = optimizer.update(grads, opt_state)
        mlp_params = optax.apply_updates(mlp_params, updates)

        # (b) advance the Markov chain
        q, _, _ = alcs_hmc_kernel(
            k_step, q, log_prob_fn, mlp_params, epsilon_base, n_leapfrog
        )
        warmup_samples.append(q)

        if print_every and (i + 1) % print_every == 0:
            print(f"  Warm-up {i+1:>4d}/{n_warmup}  |  -ESJD = {float(loss_val):.4f}")

    return mlp_params, jnp.stack(warmup_samples)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Public API  —  run_alcs_hmc
#     Signature mirrors run_hmc for drop-in comparison
# ─────────────────────────────────────────────────────────────────────────────

def run_alcs_hmc(
    key,
    log_prob_fn,
    initial_position,
    step_size:   float,
    n_leapfrog:  int,
    n_samples:   int,
    n_warmup:    int   = 500,
    lr:          float = 3e-3,
    hidden_dim:  int   = 8,
):
    """
    Run ALCS-HMC: Adaptive Local-Curvature Step-size HMC.

    Two-phase algorithm:
      Phase 1 — Warm-up (n_warmup steps):
          A tiny MLP is trained to adapt the leapfrog step size using
          gradient signals from the target.  Objective: maximise ESJD.

      Phase 2 — Sampling (n_samples steps):
          Standard HMC with the frozen (trained) MLP controlling ε.

    Args:
        key:              JAX random key
        log_prob_fn:      Log-probability function   θ → scalar
        initial_position: Starting point  (D,)
        step_size:        Base leapfrog step size  ε_base
        n_leapfrog:       Leapfrog steps per proposal
        n_samples:        Number of post-warmup samples to collect
        n_warmup:         Warm-up length for MLP training   [default 500]
        lr:               Adam learning rate for MLP         [default 3e-3]
        hidden_dim:       MLP hidden layer width             [default 8]

    Returns:
        samples:         Array  (n_samples, D)
        acceptance_rate: Mean acceptance probability over the sampling phase
    """
    key_warmup, key_sample = jr.split(key)

    # ── Phase 1: train the controller ────────────────────────────────────
    print(f"[ALCS-HMC] Warm-up phase: {n_warmup} steps, lr={lr}, hidden_dim={hidden_dim}")
    mlp_params, _ = warmup_train_controller(
        key_warmup,
        log_prob_fn,
        initial_position,
        step_size,
        n_leapfrog,
        n_warmup=n_warmup,
        lr=lr,
        hidden_dim=hidden_dim,
    )
    print("[ALCS-HMC] Warm-up complete. Weights frozen — starting sampling phase.")

    # ── Phase 2: sampling with frozen MLP ────────────────────────────────
    @jax.jit
    def one_step(q, key):
        q_new, _accepted, acc_prob = alcs_hmc_kernel(
            key, q, log_prob_fn, mlp_params, step_size, n_leapfrog
        )
        return q_new, (q_new, acc_prob)

    keys = jr.split(key_sample, n_samples)
    _, (samples, acc_probs) = jax.lax.scan(one_step, initial_position, keys)

    return samples, acc_probs.mean()


# =============================================================================
# Usage (paste below the HMC baseline in your notebook)
# =============================================================================
#
# key = jr.PRNGKey(42)
# key1, key2, key3 = jr.split(key, 3)
# initial_pos = jnp.array([0.0, 0.0])
# n_samples   = 50_000
#
# ── Rosenbrock ──────────────────────────────────────────────────────────────
# alcs_samples_rb, alcs_acc_rb = run_alcs_hmc(
#     key1, log_prob_rosenbrock, initial_pos,
#     step_size=0.2, n_leapfrog=10, n_samples=n_samples,
#     n_warmup=500, lr=3e-3,
# )
# print(f"ALCS-HMC (Rosenbrock) acceptance rate: {alcs_acc_rb:.2%}")
#
# ── Neal's Funnel ────────────────────────────────────────────────────────────
# alcs_samples_fn, alcs_acc_fn = run_alcs_hmc(
#     key2, log_prob_funnel, initial_pos,
#     step_size=0.05, n_leapfrog=10, n_samples=n_samples,
#     n_warmup=800, lr=3e-3,
# )
# print(f"ALCS-HMC (Funnel) acceptance rate: {alcs_acc_fn:.2%}")
#
# ── Diagnostics (same as baseline) ──────────────────────────────────────────
# alcs_idata = samples_to_inference_data(alcs_samples_rb, var_names=["x", "y"])
# summarize_sampler(alcs_samples_rb, "ALCS-HMC — Rosenbrock", var_names=["x","y"])
#
# plot_samples_comparison(
#     hmc_samples, alcs_samples_rb,
#     f"HMC (acc={hmc_acc:.1%})",
#     f"ALCS-HMC (acc={alcs_acc_rb:.1%})",
#     log_prob_rosenbrock, xlim=(-2, 3), ylim=(-1, 5),
# )
#
# ── Ablation: effect of hidden_dim ──────────────────────────────────────────
# for hd in [2, 4, 8, 16]:
#     s, acc = run_alcs_hmc(
#         key3, log_prob_funnel, initial_pos,
#         step_size=0.05, n_leapfrog=10, n_samples=10_000,
#         n_warmup=300, lr=3e-3, hidden_dim=hd, 
#     )
#     idata  = samples_to_inference_data(s, ["v", "x"])
#     summ   = az.summary(idata, kind="stats")
#     ess    = summ["ess_bulk"].mean()
#     print(f"hidden_dim={hd:>2d}  acc={acc:.2%}  mean_ESS={ess:.0f}")
