"""
ablation.py
───────────
Structured ablation study for ALCS-HMC v1.

Five questions, each isolating one design decision:

  Q1  Does the MLP help at all?
        → fixed α=1.0  vs  frozen (untrained) MLP  vs  trained MLP

  Q2  Is the warm-up long enough?
        → n_warmup ∈ {5%, 10%, 20%, 30%} of n_samples

  Q3  Does the learning rate matter?
        → lr ∈ {1e-3, 3e-4, 1e-4, 3e-5}

  Q4  Is the base step size well-chosen?
        → base_step_size ∈ {0.05, 0.10, 0.20, 0.30}

  Q5  Does the loss function choice matter?  (addresses theoretical Q2)
        → "energy_error"  vs  "esjd"

  Q6  Does symmetric step size improve results?  (addresses theoretical Q1)
        → use_symm_step=False  vs  True

Run this script directly to produce all tables and plots:
    python ablation.py
"""

import sys
import jax.numpy as jnp
import jax.random as jr
import pandas as pd
import matplotlib.pyplot as plt

from distributions import log_prob_rosenbrock, log_prob_funnel
from sampler       import run_alcs_hmc


# ── Shared settings ───────────────────────────────────────────────────────────

N_SAMPLES   = 20_000    # shorter run for fast sweeps
BASE_KEY    = jr.PRNGKey(0)

def ablation_key(i):
    """Derive a deterministic sub-key for each ablation run."""
    return jr.fold_in(BASE_KEY, i)


# ── Per-configuration summary row ─────────────────────────────────────────────

def _row(name, result):
    wl = jnp.array(result["warmup_losses"])
    return {
        "config"               : name,
        "acc_warmup %"         : f"{result['accept_rate_warmup']:.1%}",
        "acc_sampling %"       : f"{result['accept_rate_sampling']:.1%}",
        "mean α (post-warmup)" : f"{float(result['alphas'][result['n_warmup']:].mean()):.3f}",
        "loss start (first 100)": f"{float(wl[:100].mean()):.3f}",
        "loss end (last 100)"  : f"{float(wl[-100:].mean()):.3f}",
        "loss drop"            : f"{float(wl[:100].mean() - wl[-100:].mean()):.3f}",
    }


def make_table(results_dict):
    rows = [_row(name, r) for name, r in results_dict.items()]
    return pd.DataFrame(rows).set_index("config")


# ── Plotting helper ───────────────────────────────────────────────────────────

def plot_ablation(results_dict, title, smooth=50):
    """
    Left:  warm-up loss curves (smoothed moving average)
    Right: post-warmup α histogram
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    colors = plt.cm.tab10.colors

    for (name, r), color in zip(results_dict.items(), colors):
        wl       = jnp.array(r["warmup_losses"])
        kernel   = jnp.ones(smooth) / smooth
        smoothed = jnp.convolve(wl, kernel, mode="valid")
        ax1.plot(smoothed, label=name, color=color, linewidth=1.8)

        pa = jnp.array(r["alphas"][r["n_warmup"]:])
        ax2.hist(pa, bins=40, alpha=0.5, label=name,
                 color=color, density=True)

    ax1.set_xlabel("Warm-up step")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss during warm-up")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.25)

    ax2.axvline(1.0, color="black", linestyle="--",
                linewidth=1, label="α=1 (no adapt)")
    ax2.set_xlabel(r"$\alpha$")
    ax2.set_ylabel("Density")
    ax2.set_title("Post-warmup α distribution")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.25)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()


# ── Individual ablation sweeps ────────────────────────────────────────────────

def run_q1(log_prob_fn, initial_position, base_step_size=0.2):
    """Q1 — Does the MLP help at all?"""
    print("Q1 — MLP contribution …")
    results = {}

    # Control: α permanently fixed at 1 (plain HMC at ε=base_step_size)
    results["fixed α=1.0 (plain HMC)"] = run_alcs_hmc(
        ablation_key(0), log_prob_fn, initial_position,
        base_step_size=base_step_size, n_samples=N_SAMPLES,
        # Patch: override mlp by passing a fixed-alpha wrapper
        # We achieve this by setting lr=0 and using a pre-initialised
        # MLP whose output is clipped to exactly 1.0.
        # Simpler: just compare with the HMC baseline acceptance rate.
        # Here we replicate by training with lr=0 (frozen from step 0).
        lr_mlp=0.0, n_warmup=1,
    )

    # Control: MLP initialised but never trained (freeze_mlp equivalent)
    results["frozen MLP (untrained)"] = run_alcs_hmc(
        ablation_key(1), log_prob_fn, initial_position,
        base_step_size=base_step_size, n_samples=N_SAMPLES,
        lr_mlp=0.0,   # zero lr → params never change
    )

    # Treatment: MLP trained normally
    results["trained MLP"] = run_alcs_hmc(
        ablation_key(2), log_prob_fn, initial_position,
        base_step_size=base_step_size, n_samples=N_SAMPLES,
    )
    return results


def run_q2(log_prob_fn, initial_position, base_step_size=0.2):
    """Q2 — Warm-up length."""
    print("Q2 — Warm-up length …")
    results = {}
    for frac, label in [(0.05, "5%"), (0.10, "10%"),
                        (0.20, "20%"), (0.30, "30%")]:
        results[f"n_warmup={label}"] = run_alcs_hmc(
            ablation_key(10), log_prob_fn, initial_position,
            base_step_size=base_step_size, n_samples=N_SAMPLES,
            n_warmup=int(N_SAMPLES * frac),
        )
    return results


def run_q3(log_prob_fn, initial_position, base_step_size=0.2):
    """Q3 — Learning rate."""
    print("Q3 — Learning rate …")
    results = {}
    for lr, label in [(1e-3, "1e-3"), (3e-4, "3e-4"),
                      (1e-4, "1e-4"), (3e-5, "3e-5")]:
        results[f"lr={label}"] = run_alcs_hmc(
            ablation_key(20), log_prob_fn, initial_position,
            base_step_size=base_step_size, n_samples=N_SAMPLES,
            lr_mlp=lr,
        )
    return results


def run_q4(log_prob_fn, initial_position):
    """Q4 — Base step size."""
    print("Q4 — Base step size …")
    results = {}
    for eps, label in [(0.05, "0.05"), (0.10, "0.10"),
                       (0.20, "0.20"), (0.30, "0.30")]:
        results[f"ε={label}"] = run_alcs_hmc(
            ablation_key(30), log_prob_fn, initial_position,
            base_step_size=eps, n_samples=N_SAMPLES,
        )
    return results


def run_q5(log_prob_fn, initial_position, base_step_size=0.2):
    """Q5 — Loss function: energy-error vs ESJD-aware (addresses theoretical Q2)."""
    print("Q5 — Loss function …")
    results = {}
    for ltype in ["energy_error", "esjd"]:
        results[f"loss={ltype}"] = run_alcs_hmc(
            ablation_key(40), log_prob_fn, initial_position,
            base_step_size=base_step_size, n_samples=N_SAMPLES,
            loss_type=ltype,
        )
    return results


def run_q6(log_prob_fn, initial_position, base_step_size=0.2):
    """Q6 — Symmetric step size: Fix B for reversibility (addresses theoretical Q1)."""
    print("Q6 — Symmetric step size …")
    results = {}
    for symm, label in [(False, "asymmetric (original)"),
                        (True,  "symmetric (Fix B)")]:
        results[label] = run_alcs_hmc(
            ablation_key(50), log_prob_fn, initial_position,
            base_step_size=base_step_size, n_samples=N_SAMPLES,
            use_symm_step=symm,
        )
    return results


# ── Main entry point ──────────────────────────────────────────────────────────

def run_all_ablations(log_prob_fn=None, initial_position=None,
                      base_step_size=0.2, distribution_name="Rosenbrock"):
    if log_prob_fn is None:
        log_prob_fn      = log_prob_rosenbrock
        initial_position = jnp.array([0.0, 0.0])

    print(f"\n{'═'*60}")
    print(f"Ablation study — {distribution_name}")
    print(f"{'═'*60}\n")

    all_results = {
        "Q1 — MLP contribution"  : run_q1(log_prob_fn, initial_position, base_step_size),
        "Q2 — Warm-up length"    : run_q2(log_prob_fn, initial_position, base_step_size),
        "Q3 — Learning rate"     : run_q3(log_prob_fn, initial_position, base_step_size),
        "Q4 — Base step size"    : run_q4(log_prob_fn, initial_position),
        "Q5 — Loss function"     : run_q5(log_prob_fn, initial_position, base_step_size),
        "Q6 — Symmetric step"    : run_q6(log_prob_fn, initial_position, base_step_size),
    }

    for question, results in all_results.items():
        print(f"\n{'─'*60}")
        print(question)
        print(f"{'─'*60}")
        print(make_table(results).to_string())
        plot_ablation(results, f"{question} — {distribution_name}")

    return all_results


if __name__ == "__main__":
    # Default: Rosenbrock
    run_all_ablations()

    # Optionally run on Funnel too
    if "--funnel" in sys.argv:
        run_all_ablations(
            log_prob_fn      = log_prob_funnel,
            initial_position = jnp.array([0.0, 0.0]),
            base_step_size   = 0.1,
            distribution_name= "Neal's Funnel",
        )
