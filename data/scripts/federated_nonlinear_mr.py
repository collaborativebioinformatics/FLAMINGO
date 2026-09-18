"""Fit the non-linear causal curve: concatenated individual-level data versus summary statistics.

Concatenated route: 2SLS with X and X^2 as endogenous regressors, instrumented
by the SNP-predicted X and its square (any function of G is a valid
instrument), with site intercepts. Recovers theta1 and theta2.

Summary-statistic route: per-SNP linear GWAS effects carry only the average
slope, so the best it can do is the IVW line from federated_summary_mr.py.

Federated-learning routes (optional, for contrast), from ../federated_learning:
the NVFlare FedAvg MLP fitted naively (E[Y | X], confounded) and as a
federated 2SRI Mendelian randomization (site-local first stage X ~ SNPs, then
a federated f(X) + c * residual, whose f is the causal curve). Read from
federated_learning/results/<method>/<shape>/curves.csv when present;
--federated_methods picks which of them are drawn (default both).

Writes results/nonlinear.<shape>.png (dose-response curves) and prints the
coefficient estimates.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import polars as pl

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import flamingo_fedmr as fm

sys.path.insert(0, str(Path(__file__).parent))
from federated_summary_mr import gwas, ivw, quadratic_2sls  # noqa: E402
from simulate_basic import causal_curve  # noqa: E402

POOLED_COLOR, SUMSTATS_COLOR, FED2SLS_COLOR, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#8a2be2", "#1f1f1e", "#6b6a63", "#e6e5df"
FL_STYLE = {"naive": ("#1baf7a", "federated learning, naive MLP: E[Y | X], no instruments (confounded)"),
            "2sri": ("#4a3aa7", "federated Fed-2SRI (FedAvg-trained network): f(X), first-stage residual as control function")}


def load_sites(folder: Path) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    sites = []
    for csv in sorted(folder.glob("site*.csv")):
        df = pl.read_csv(csv)
        sites.append((df.select(pl.col("^snp.*$")).to_numpy().astype(float), df["X"].to_numpy(), df["Y"].to_numpy()))
    return sites


def sumstats_slope(sites) -> tuple[float, float]:
    """Meta-analysis of per-site IVW slopes, exactly as the sumstats route computes it."""
    est, w = [], []
    for G, X, Y in sites:
        bx, _ = gwas(G, X)
        by, se_y = gwas(G, Y)
        e, s = ivw(bx, by, se_y)
        est.append(e); w.append(1 / s**2)
    est, w = np.array(est), np.array(w)
    return float(np.sum(w * est) / np.sum(w)), float(np.sqrt(1 / np.sum(w)))


def federated_curve(path: Path, x: np.ndarray) -> tuple[np.ndarray, int] | None:
    """Last-round global-model f(X) from the NVFlare run, centred at X = 0 and
    interpolated onto x. Returns (curve, round) or None if the file is missing."""
    if not path.exists():
        return None
    df = pl.read_csv(path)
    last = df["round"].max()
    c = df.filter(pl.col("round") == last).group_by("x").agg(pl.col("f").mean()).sort("x")
    xs, fs = c["x"].to_numpy(), c["f"].to_numpy()
    fs = fs - np.interp(0.0, xs, fs)
    return np.interp(x, xs, fs), int(last)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--shape", choices=["quadratic", "threshold"], default="quadratic")
    p.add_argument("--sites", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--federated", type=Path, default=None,
                   help="results/ root of the NVFlare runs (default: ../federated_learning/results)")
    p.add_argument("--federated_methods", nargs="*", choices=list(FL_STYLE), default=list(FL_STYLE),
                   help="which federated-learning curves to draw when present (default: all; pass none to draw no curve)")
    a = p.parse_args()
    a.federated = a.federated or Path(__file__).resolve().parents[2] / "federated_learning" / "results"
    a.sites = a.sites or Path("simulated_data/federated") / a.shape
    a.out = a.out or Path("results") / f"nonlinear.{a.shape}.png"
    manifest = json.loads((a.sites / "manifest.json").read_text())
    t1, t2 = manifest["theta1"], manifest["theta2"]

    sites = load_sites(a.sites)
    theta, cov = quadratic_2sls(sites)
    se = np.sqrt(np.diag(cov))
    slope, slope_se = sumstats_slope(sites)
    fmr = fm.LocalFirstStageFedMR(basis="quadratic", robust=False).run(
        [fm.SiteData(f"site{k + 1:02d}", G, X, Y) for k, (G, X, Y) in enumerate(sites)]).result
    fm_theta = np.array([fmr["X"], fmr["X2"]])
    fm_diff = float(np.max(np.abs(fm_theta - theta)))

    print(f"{a.shape} set, true theta1={t1}, theta2={t2}")
    print(f"concatenated quadratic 2SLS:  theta1 {theta[0]:.3f} ({se[0]:.3f})   theta2 {theta[1]:.3f} ({se[1]:.3f})")
    print(f"sumstats linear IVW:          slope  {slope:.3f} ({slope_se:.3f})   theta2 not identifiable")
    print(f"federated Fed-2SLS quadratic:    theta1 {fm_theta[0]:.3f} ({fmr.se('X'):.3f})   theta2 {fm_theta[1]:.3f} "
          f"({fmr.se('X2'):.3f})   |diff from concatenated| = {fm_diff:.1e}")

    # dose-response curves, centred so every curve passes through f(0) = 0
    x = np.linspace(-2.5, 2.5, 200)
    truth = causal_curve(a.shape, x, t1, t2)
    basis = np.column_stack([x, x**2])
    fit = basis @ theta
    fit_se = np.sqrt(np.einsum("ij,jk,ik->i", basis, cov, basis))
    line = slope * x
    fl = {m: federated_curve(a.federated / m / a.shape / "curves.csv", x) for m in a.federated_methods}

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.fill_between(x, fit - 1.96 * fit_se, fit + 1.96 * fit_se, color=POOLED_COLOR, alpha=0.18, linewidth=0)
    ax.plot(x, truth, color=INK, linewidth=2, linestyle="--", label=f"true curve ({a.shape}, θ1={t1}, θ2={t2})")
    ax.plot(x, fit, color=POOLED_COLOR, linewidth=2,
            label=f"concatenated: quadratic 2SLS  θ1={theta[0]:.2f}, θ2={theta[1]:.2f} (95% band)")
    ax.plot(x, line, color=SUMSTATS_COLOR, linewidth=2, label=f"sumstats: linear IVW  slope={slope:.2f}")
    ax.plot(x, basis @ fm_theta, color=FED2SLS_COLOR, linewidth=1.4, linestyle=(0, (1, 2)),
            label=f"federated: Fed-2SLS sufficient statistics, identical to concatenated (|Δθ| = {fm_diff:.0e})")
    for m, (color, label) in FL_STYLE.items():
        if m not in fl:
            continue
        if fl[m] is None:
            print(f"no federated {m} curve under {a.federated}; run federated_learning/job.py "
                  f"--dataset {a.shape} --method {m} to add it")
            continue
        ax.plot(x, fl[m][0], color=color, linewidth=2, linestyle=(0, (5, 2)), label=f"{label}, round {fl[m][1]}")
        print(f"federated {m} curve:          from {a.federated / m / a.shape}")
    X_all = np.concatenate([s[1] for s in sites])
    ax2 = ax.twinx()
    ax2.hist(X_all, bins=60, color=GRID, alpha=0.6, zorder=0)
    ax2.set_ylim(0, ax2.get_ylim()[1] * 4)
    ax2.set_yticks([]); ax2.spines[:].set_visible(False)
    ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
    ax.set_xlabel("exposure X (standardised)")
    ax.set_ylabel("causal effect on Y, relative to X = 0")
    ax.set_title(f"{a.shape} model: recovering the dose-response curve (n = {len(X_all):,}, {len(sites)} sites)",
                 loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(a.out)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
