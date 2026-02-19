"""
diagnostics.py
──────────────
Plotting and ArviZ diagnostic utilities for ALCS-HMC.
"""

import arviz as az
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from mlp import mlp_forward


# ── ArviZ helpers ─────────────────────────────────────────────────────────────

def samples_to_inference_data(samples, var_names=None):
    """
    Convert (n_samples, D) array to ArviZ InferenceData.
    ArviZ expects shape (n_chains, n_samples) per variable.
    """
    if var_names is None:
        var_names = [f"theta_{i}" for i in range(samples.shape[1])]
    data_dict = {name: samples[None, :, i]
                 for i, name in enumerate(var_names)}
    return az.convert_to_inference_data(data_dict)


def summarize_sampler(samples, name, var_names=None):
    """Print ArviZ summary (mean, sd, HDI) for a sample array."""
    idata = samples_to_inference_data(samples, var_names)
    print(f"\n=== {name} ===")
    from IPython.display import display
    display(az.summary(idata, kind="stats"))
    

def score_against_truth(samples, true_moments, var_names):
    """
    Print a table comparing sample moments to known ground truth.

    true_moments: dict of {var_name: {"mean": ..., "sd": ...}}
    """
    print(f"\n{'─'*55}")
    print(f"{'Variable':<10} {'Truth mean':>12} {'Sample mean':>12} "
          f"{'Truth sd':>10} {'Sample sd':>10}")
    print(f"{'─'*55}")
    for i, vn in enumerate(var_names):
        tm = true_moments[vn]["mean"]
        ts = true_moments[vn]["sd"]
        sm = float(samples[:, i].mean())
        ss = float(samples[:, i].std())
        print(f"{vn:<10} {tm:>12.3f} {sm:>12.3f} {ts:>10.3f} {ss:>10.3f}")
    print(f"{'─'*55}")


# ── Distribution visualisation ────────────────────────────────────────────────

def plot_distribution(log_prob_fn, title, xlim=(-4, 4), ylim=(-4, 4)):
    x = jnp.linspace(*xlim, 200)
    y = jnp.linspace(*ylim, 200)
    X, Y = jnp.meshgrid(x, y)
    positions  = jnp.stack([X.ravel(), Y.ravel()], axis=-1)
    log_probs  = jax.vmap(log_prob_fn)(positions).reshape(X.shape)

    plt.figure(figsize=(6, 5))
    plt.contourf(X, Y, jnp.exp(log_probs), levels=50, cmap="viridis")
    plt.colorbar(label="Probability density")
    plt.xlabel(r"$x$"); plt.ylabel(r"$y$")
    plt.title(title)
    plt.tight_layout(); plt.show()


# ── Sample scatter comparison ─────────────────────────────────────────────────

def plot_samples_comparison(samples1, samples2, label1, label2,
                             log_prob_fn, xlim, ylim):
    x = jnp.linspace(*xlim, 100)
    y = jnp.linspace(*ylim, 100)
    X, Y      = jnp.meshgrid(x, y)
    positions = jnp.stack([X.ravel(), Y.ravel()], axis=-1)
    log_probs = jax.vmap(log_prob_fn)(positions).reshape(X.shape)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, samples, label in zip(axes, [samples1, samples2], [label1, label2]):
        ax.contour(X, Y, jnp.exp(log_probs), levels=10,
                   colors="gray", alpha=0.5)
        ax.scatter(samples[::5, 0], samples[::5, 1],
                   alpha=0.3, s=5, c="blue")
        ax.set_xlabel(r"$\theta_0$"); ax.set_ylabel(r"$\theta_1$")
        ax.set_title(label)
        ax.set_xlim(xlim); ax.set_ylim(ylim)
    plt.tight_layout(); plt.show()


# ── Learned controller visualisation ─────────────────────────────────────────

def plot_learned_controller(mlp_params, base_step_size,
                             title="Learned Step-size Controller"):
    log_norms = jnp.linspace(-3.0, 5.0, 300)
    alphas    = jax.vmap(lambda x: mlp_forward(mlp_params, x))(log_norms)
    eff_steps = base_step_size * alphas

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

    ax1.plot(log_norms, alphas, color="#E05A2B", linewidth=2.5)
    ax1.axhline(1.0, color="#888", linestyle="--", linewidth=1,
                label=r"$\alpha=1$ (no adapt.)")
    ax1.fill_between(log_norms, 1.0, alphas, alpha=0.15, color="#E05A2B")
    ax1.set_xlabel(r"$\log\|\nabla E(\mathbf{q})\|$")
    ax1.set_ylabel(r"$\alpha$"); ax1.set_title("MLP output")
    ax1.legend(); ax1.grid(alpha=0.25)

    ax2.plot(log_norms, eff_steps, color="#2B7BE0", linewidth=2.5)
    ax2.axhline(base_step_size, color="#888", linestyle="--", linewidth=1,
                label=rf"$\varepsilon_{{base}}={base_step_size}$")
    ax2.fill_between(log_norms, base_step_size, eff_steps,
                     alpha=0.15, color="#2B7BE0")
    ax2.set_xlabel(r"$\log\|\nabla E(\mathbf{q})\|$")
    ax2.set_ylabel(r"$\varepsilon_{\rm eff}$")
    ax2.set_title("Effective step size")
    ax2.legend(); ax2.grid(alpha=0.25)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()


# ── Alpha trajectory ──────────────────────────────────────────────────────────

def plot_alpha_trajectories(results_list, n_warmup, title="α trajectory"):
    """
    results_list: list of (label, alphas_array) tuples
    """
    fig, ax = plt.subplots(figsize=(13, 3.5))
    colors  = plt.cm.tab10.colors
    for (label, alphas), color in zip(results_list, colors):
        ax.plot(alphas, alpha=0.6, linewidth=0.6, color=color, label=label)
    ax.axvline(n_warmup, color="#E05A2B", linestyle="--",
               linewidth=1.5, label="warm-up end")
    ax.set_xlabel("Sample index")
    ax.set_ylabel(r"$\alpha$")
    ax.set_title(title)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.25)
    plt.tight_layout(); plt.show()
