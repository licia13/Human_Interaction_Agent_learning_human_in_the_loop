"""
SAC vs AWAC — Training Comparison
-----------------------------------
Loads the metrics CSVs produced by policy_learn.py and policy_learn_awac.py
and generates a 6-panel comparison figure saved to saved/comparison.png.

Run from the repository root:
    python -m scripts.compare_policies

The script works even if only one algorithm has finished training — it simply
skips any missing CSV and labels what is available.

Panels
------
1. Success Rate           — the primary evaluation metric
2. Smoothed Success Rate  — EMA-smoothed for trend visibility
3. Episode Reward         — trajectory-level reward (dot(w, φ(τ)))
4. Episode Length         — how many steps the robot takes per episode
5. Actor Loss             — policy optimisation cost
6. Critic Loss            — value-function TD error
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")           # headless — no display required
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_csv(path: Path) -> Optional[dict]:
    """Return a dict of column_name -> list[float], or None if file missing."""
    if not path.exists():
        return None
    data: dict = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, val in row.items():
                data.setdefault(key, []).append(
                    float(val) if val not in ("", "nan") else np.nan
                )
    return {k: np.array(v, dtype=float) for k, v in data.items()} if data else None


def ema(values: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """Exponential moving average — keeps the same length as the input."""
    out = np.empty_like(values)
    out[0] = values[0]
    for i in range(1, len(values)):
        if np.isnan(values[i]):
            out[i] = out[i - 1]
        else:
            out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out


def _plot_metric(
    ax: plt.Axes,
    sac: Optional[dict],
    awac: Optional[dict],
    column: str,
    title: str,
    ylabel: str,
    smooth: bool = False,
    alpha_raw: float = 0.25,
) -> None:
    """Plot one metric on the given axes for both algorithms."""
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Environment Steps", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)

    plotted_any = False
    for label, data, color in [("SAC", sac, "#2196F3"), ("AWAC", awac, "#FF5722")]:
        if data is None or column not in data:
            continue
        x = data["steps"]
        y = data[column]
        valid = ~np.isnan(y)
        if not valid.any():
            continue
        plotted_any = True
        if smooth:
            ax.plot(x[valid], y[valid], color=color, alpha=alpha_raw, linewidth=0.8)
            ax.plot(x[valid], ema(y[valid]), color=color, linewidth=2.0, label=label)
        else:
            ax.plot(x[valid], y[valid], color=color, linewidth=1.5, label=label)

    if plotted_any:
        ax.legend(fontsize=8)


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    saved_dir = repo_root / "saved"

    sac_data  = load_csv(saved_dir / "metrics_sac.csv")
    awac_data = load_csv(saved_dir / "metrics_awac.csv")

    if sac_data is None and awac_data is None:
        print("No metrics CSVs found yet. Run policy_learn.py and/or "
              "policy_learn_awac.py first.")
        return

    if sac_data is None:
        print("metrics_sac.csv not found — showing AWAC only.")
    if awac_data is None:
        print("metrics_awac.csv not found — showing SAC only.")

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle("SAC vs AWAC — Training Comparison", fontsize=14, fontweight="bold", y=1.01)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]

    # Panel 1 — raw success rate
    _plot_metric(axes[0], sac_data, awac_data,
                 column="success_rate",
                 title="① Success Rate",
                 ylabel="Success Rate (10 runs)",
                 smooth=False)

    # Panel 2 — EMA-smoothed success rate (better for spotting trends)
    _plot_metric(axes[1], sac_data, awac_data,
                 column="success_rate",
                 title="② Success Rate (EMA smoothed)",
                 ylabel="Success Rate",
                 smooth=True)

    # Panel 3 — episode reward
    _plot_metric(axes[2], sac_data, awac_data,
                 column="ep_reward",
                 title="③ Episode Reward  (dot(w, φ(τ)))",
                 ylabel="Trajectory Reward",
                 smooth=True)

    # Panel 4 — episode length
    _plot_metric(axes[3], sac_data, awac_data,
                 column="ep_length",
                 title="④ Episode Length",
                 ylabel="Steps per Episode",
                 smooth=True)

    # Panel 5 — actor loss
    _plot_metric(axes[4], sac_data, awac_data,
                 column="actor_loss",
                 title="⑤ Actor Loss",
                 ylabel="Loss",
                 smooth=True)

    # Panel 6 — critic loss
    _plot_metric(axes[5], sac_data, awac_data,
                 column="critic_loss",
                 title="⑥ Critic Loss  (TD error)",
                 ylabel="Loss",
                 smooth=True)

    plt.tight_layout()
    out_path = saved_dir / "comparison.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Comparison figure saved to {out_path}")

    # ── Print a quick summary table ───────────────────────────────────────────
    print("\n── Final checkpoint summary ──────────────────────────────────────")
    for label, data in [("SAC ", sac_data), ("AWAC", awac_data)]:
        if data is None:
            print(f"  {label}: no data")
            continue
        final_sr = data["success_rate"][~np.isnan(data["success_rate"])]
        best_sr  = float(np.max(final_sr)) if len(final_sr) else float("nan")
        last_sr  = float(final_sr[-1])     if len(final_sr) else float("nan")
        n_evals  = len(final_sr)
        print(f"  {label}: {n_evals} eval checkpoints | "
              f"best={best_sr:.2f}  last={last_sr:.2f}")


if __name__ == "__main__":
    main()
