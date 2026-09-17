"""Repeat the basic simulation over many seeds and compare naive OLS with 2SLS.

Writes results/sweep.csv (one row per seed) and results/sweep.png.
"""

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import polars as pl

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from simulate_basic import simulate, simulate_nonlinear  # noqa: E402

OLS_COLOR, TSLS_COLOR = "#eb6834", "#2a78d6"
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e6e5df"


def estimate(df):
    G = df.select(pl.col("^snp.*$")).to_numpy().astype(float)
    X, Y = df["X"].to_numpy(), df["Y"].to_numpy()
    ols = np.cov(X, Y)[0, 1] / np.var(X, ddof=1)
    Z = np.column_stack([np.ones(len(X)), G])
    Xhat = Z @ np.linalg.lstsq(Z, X, rcond=None)[0]
    tsls = np.cov(Xhat, Y)[0, 1] / np.var(Xhat, ddof=1)
    return ols, tsls


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seeds", type=int, default=300)
    p.add_argument("--n", type=int, default=10_000)
    p.add_argument("--n-snps", type=int, default=20)
    p.add_argument("--shape", choices=["linear", "quadratic", "threshold"], default="linear")
    p.add_argument("--theta", type=float, default=0.3, help="theta, or theta1 when non-linear")
    p.add_argument("--theta2", type=float, default=0.0)
    p.add_argument("--h2-x", type=float, default=0.10)
    p.add_argument("--gamma-x", type=float, default=0.3)
    p.add_argument("--gamma-y", type=float, default=0.3)
    p.add_argument("--out", type=Path, default=Path("results/sweep"))
    a = p.parse_args()

    rows = []
    for seed in range(1, a.seeds + 1):
        if a.shape == "linear":
            df, truth = simulate(a.n, a.n_snps, a.theta, a.h2_x, a.gamma_x, a.gamma_y, seed)
            target = a.theta
        else:
            df, truth = simulate_nonlinear(a.n, a.n_snps, a.shape, a.theta, a.theta2,
                                           a.h2_x, a.gamma_x, a.gamma_y, seed)
            target = truth["avg_slope"]
        ols, tsls = estimate(df)
        rows.append({"seed": seed, "ols": ols, "tsls": tsls, "avg_slope": target})
    res = pl.DataFrame(rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    res.write_csv(a.out.with_suffix(".csv"))

    ols, tsls = res["ols"].to_numpy(), res["tsls"].to_numpy()
    target = float(res["avg_slope"].mean())
    for name, v in (("OLS", ols), ("2SLS", tsls)):
        print(f"{name:5s} mean {v.mean():.3f}  sd {v.std(ddof=1):.3f}  bias vs avg slope {v.mean() - target:+.3f}")

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    bins = np.linspace(min(ols.min(), tsls.min()), max(ols.max(), tsls.max()), 40)
    ax.hist(tsls, bins, color=TSLS_COLOR, alpha=0.85, edgecolor="white", linewidth=0.5,
            label=f"2SLS through SNPs (MR)   mean {tsls.mean():.3f}, sd {tsls.std(ddof=1):.3f}")
    ax.hist(ols, bins, color=OLS_COLOR, alpha=0.85, edgecolor="white", linewidth=0.5,
            label=f"naive OLS of Y on X   mean {ols.mean():.3f}, sd {ols.std(ddof=1):.3f}")
    ax.axvline(target, color=INK, linewidth=1.2, linestyle="--")
    line_label = f"true θ = {a.theta}" if a.shape == "linear" else f"true average slope = {target:.3f}"
    ax.text(target + 0.002, ax.get_ylim()[1] * 0.6, line_label, ha="left", va="center",
            color=INK, fontsize=9, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none"))
    ax.set_xlabel("estimated causal effect of X on Y")
    ax.set_ylabel("number of seeds")
    model = "linear" if a.shape == "linear" else f"{a.shape}, θ1={a.theta}, θ2={a.theta2}"
    ax.set_title(f"{model}  ·  {a.seeds} seeds, n={a.n:,}, {a.n_snps} SNPs, SNP R²={a.h2_x}",
                 fontsize=11, loc="left", color=INK)
    ax.legend(frameon=False, loc="upper left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(a.out.with_suffix(".png"))
    print(f"wrote {a.out}.csv and {a.out}.png")


if __name__ == "__main__":
    main()
