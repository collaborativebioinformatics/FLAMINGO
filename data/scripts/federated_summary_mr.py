"""Federated MR from per-site GWAS summary statistics, for each phenotype model.

Each site runs its own GWAS of X and of Y on its SNPs (one-sample: same people
for both) and shares only per-SNP effects and standard errors. For the Cox
survival model the "GWAS of Y" is a per-SNP Cox regression, so beta_y is a
log hazard ratio per allele and the ratio estimate is on the log-HR scale. From those
summary statistics this script computes, per site, the inverse-variance
weighted (IVW) MR estimate of the causal slope, then combines the sites by
inverse-variance meta-analysis. SNPs are site-specific draws, so per-SNP
effects are never meta-analysed across sites; only the causal estimates are.

For comparison it also concatenates the individual-level data and fits one
pooled 2SLS: first stage per site (each site's SNPs instrument only its own
people), second stage across everyone with site intercepts. That is the
benchmark the federated estimate would match if data could be shared.

Writes simulated_data/federated/<shape>/sumstats/<site>.sumstats.csv, a
per-site table results/sumstats.<shape>.csv, and a forest plot
results/sumstats.<shape>.png.
"""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import polars as pl
from lifelines import CoxPHFitter

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SITE_COLOR, META_COLOR, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1f1f1e", "#6b6a63", "#e6e5df"


def gwas(G, y):
    """Per-SNP simple linear regression of y on each column of G. Returns (beta, se)."""
    Gc = G - G.mean(axis=0)
    yc = y - y.mean()
    sxx = (Gc**2).sum(axis=0)
    beta = (Gc * yc[:, None]).sum(axis=0) / sxx
    resid = yc[:, None] - Gc * beta
    n = len(y)
    sigma2 = (resid**2).sum(axis=0) / (n - 2)
    return beta, np.sqrt(sigma2 / sxx)


def cox_gwas(G, time, event):
    """Per-SNP Cox regression of (time, event) on each column of G. Returns (log HR, se)."""
    beta, se = np.empty(G.shape[1]), np.empty(G.shape[1])
    for j in range(G.shape[1]):
        d = pd.DataFrame({"time": time, "event": event, "g": G[:, j]})
        cph = CoxPHFitter().fit(d, "time", "event")
        beta[j], se[j] = cph.params_["g"], cph.standard_errors_["g"]
    return beta, se


def ivw(bx, by, se_y):
    """Inverse-variance weighted MR slope through the origin, with its standard error."""
    w = 1.0 / se_y**2
    est = np.sum(w * bx * by) / np.sum(w * bx**2)
    se = np.sqrt(1.0 / np.sum(w * bx**2))
    return est, se


def _first_stage(sites):
    """Per-site first stage: each site's SNPs predict only its own people."""
    xhat, X, Y, site_idx = [], [], [], []
    for k, (G, x, y) in enumerate(sites):
        Z = np.column_stack([np.ones(len(x)), G])
        xhat.append(Z @ np.linalg.lstsq(Z, x, rcond=None)[0])
        X.append(x); Y.append(y); site_idx.append(np.full(len(x), k))
    return tuple(map(np.concatenate, (xhat, X, Y, site_idx)))


def pooled_2sps_cox(sites):
    """Concatenate all sites and fit one Cox model on the SNP-predicted X, stratified by site."""
    xhat, X, Y, site_idx = _first_stage(sites)
    d = pd.DataFrame({"time": Y[:, 0], "event": Y[:, 1], "Xhat": xhat, "X": X, "site": site_idx})
    two_stage = CoxPHFitter().fit(d[["time", "event", "Xhat", "site"]], "time", "event", strata=["site"])
    naive = CoxPHFitter().fit(d[["time", "event", "X", "site"]], "time", "event", strata=["site"])
    return (float(two_stage.params_["Xhat"]), float(two_stage.standard_errors_["Xhat"]),
            float(naive.params_["X"]))


def pooled_2sls(sites: list[tuple[np.ndarray, np.ndarray, np.ndarray]]):
    """Concatenate all sites and fit one 2SLS with site-specific first stages and site intercepts."""
    xhat, X, Y, site_idx = _first_stage(sites)
    D = np.column_stack([xhat, np.eye(len(sites))[site_idx]])
    coef, *_ = np.linalg.lstsq(D, Y, rcond=None)
    # 2SLS standard error: residuals from the structural equation, design from the fitted stage
    D_struct = np.column_stack([X, np.eye(len(sites))[site_idx]])
    resid = Y - D_struct @ coef
    sigma2 = resid @ resid / (len(Y) - D.shape[1])
    se = np.sqrt(sigma2 * np.linalg.inv(D.T @ D)[0, 0])
    naive = np.linalg.lstsq(D_struct, Y, rcond=None)[0][0]
    return float(coef[0]), float(se), float(naive)


def site_summary(csv: Path, sumstats_dir: Path, survival: bool) -> dict:
    df = pl.read_csv(csv)
    G = df.select(pl.col("^snp.*$")).to_numpy().astype(float)
    X = df["X"].to_numpy()
    bx, se_x = gwas(G, X)
    if survival:
        time, event = df["time"].to_numpy(), df["event"].to_numpy()
        site_summary.raw.append((G, X, np.column_stack([time, event])))
        by, se_y = cox_gwas(G, time, event)
        naive_d = pd.DataFrame({"time": time, "event": event, "X": X})
        naive = float(CoxPHFitter().fit(naive_d, "time", "event").params_["X"])
    else:
        Y = df["Y"].to_numpy()
        site_summary.raw.append((G, X, Y))
        by, se_y = gwas(G, Y)
        naive = float(np.cov(X, Y)[0, 1] / np.var(X, ddof=1))
    site = csv.stem
    pl.DataFrame({"snp": [f"snp{j}" for j in range(G.shape[1])], "n": df.height,
                  "beta_x": bx, "se_x": se_x, "beta_y": by, "se_y": se_y}
                 ).write_csv(sumstats_dir / f"{site}.sumstats.csv")

    est, se = ivw(bx, by, se_y)
    f_stat = float(np.mean((bx / se_x) ** 2))
    truth = json.loads(csv.with_suffix(".truth.json").read_text())
    return {"site": site, "n": df.height, "avg_slope": truth["avg_slope"],
            "ivw": est, "ivw_se": se, "naive_ols": naive, "mean_F": f_stat}


site_summary.raw = []


def forest(res, meta, meta_se, pooled, pooled_se, pooled_naive, target, target_label, shape, out: Path):
    sites = res["site"].to_list()
    y = np.arange(len(sites))[::-1]
    fig, ax = plt.subplots(figsize=(8, 6.4), dpi=150)
    ax.axvline(target, color=INK, linewidth=1.2, linestyle="--")
    ax.errorbar(res["ivw"], y, xerr=1.96 * res["ivw_se"], fmt="o", color=SITE_COLOR, ms=6,
                ecolor=SITE_COLOR, elinewidth=2, capsize=0, label="site IVW (95% CI)")
    ax.scatter(res["naive_ols"], y, marker="|", s=120, color=MUTED, linewidths=2, label="site naive OLS", zorder=3)
    ax.errorbar([meta], [-1], xerr=[1.96 * meta_se], fmt="D", color=META_COLOR, ms=8,
                ecolor=META_COLOR, elinewidth=3, label="sumstats: meta-analysis of site IVW")
    pooled_label = "concatenated: one stratified 2SPS Cox" if shape == "cox" else "concatenated: one pooled 2SLS"
    ax.errorbar([pooled], [-2], xerr=[1.96 * pooled_se], fmt="s", color=INK, ms=7,
                ecolor=INK, elinewidth=3, label=pooled_label)
    ax.scatter([pooled_naive], [-2], marker="|", s=120, color=MUTED, linewidths=2, zorder=3)
    ax.axhline(-0.5, color=GRID, linewidth=0.8)
    n_all = res["n"].sum()
    ax.set_yticks(list(y) + [-1, -2])
    ax.set_yticklabels([f"{s}  (n={n:,}, F={f:.0f})" for s, n, f in zip(sites, res["n"], res["mean_F"])]
                       + [f"sumstats  (n={n_all:,})", f"concatenated  (n={n_all:,})"], fontsize=9)
    ax.set_xlabel("estimated log hazard ratio per unit X" if shape == "cox" else "estimated causal effect of X on Y")
    ax.set_title(f"{shape} model: MR from per-site GWAS summary statistics\ndashed line: {target_label}",
                 loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--shape", choices=["linear", "quadratic", "threshold", "cox"], default="linear")
    p.add_argument("--sites", type=Path, default=None, help="default simulated_data/federated/<shape>")
    p.add_argument("--out", type=Path, default=None, help="output stem; default results/sumstats.<shape>")
    a = p.parse_args()
    a.sites = a.sites or Path("simulated_data/federated") / a.shape
    a.out = a.out or Path("results") / f"sumstats.{a.shape}"

    manifest = json.loads((a.sites / "manifest.json").read_text())
    sumstats_dir = a.sites / "sumstats"
    sumstats_dir.mkdir(exist_ok=True)
    site_summary.raw = []
    rows = [site_summary(csv, sumstats_dir, a.shape == "cox") for csv in sorted(a.sites.glob("site*.csv"))]
    res = pl.DataFrame(rows)

    # what a linear estimator targets: theta for linear/cox, the n-weighted mean of
    # each site's population-average slope for the non-linear curves
    n_arr, slope_arr = res["n"].to_numpy(), res["avg_slope"].to_numpy()
    target = float(np.sum(n_arr * slope_arr) / np.sum(n_arr))
    target_label = (f"true θ = {manifest['theta1']}" if a.shape in ("linear", "cox")
                    else f"true average slope = {target:.3f}  (θ1={manifest['theta1']}, θ2={manifest['theta2']})")

    w = 1.0 / res["ivw_se"].to_numpy() ** 2
    meta = float(np.sum(w * res["ivw"].to_numpy()) / np.sum(w))
    meta_se = float(np.sqrt(1.0 / np.sum(w)))
    q = float(np.sum(w * (res["ivw"].to_numpy() - meta) ** 2))

    pooled_fn = pooled_2sps_cox if a.shape == "cox" else pooled_2sls
    pooled, pooled_se, pooled_naive = pooled_fn(site_summary.raw)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    csv_out, png_out = Path(f"{a.out}.csv"), Path(f"{a.out}.png")
    res.write_csv(csv_out)
    forest(res, meta, meta_se, pooled, pooled_se, pooled_naive, target, target_label, a.shape, png_out)

    with pl.Config(tbl_rows=-1, float_precision=3):
        print(res)
    print(f"\nmeta IVW  {meta:.3f}  se {meta_se:.3f}  95% CI [{meta - 1.96 * meta_se:.3f}, {meta + 1.96 * meta_se:.3f}]")
    print(f"pooled {'2SPS Cox' if a.shape == 'cox' else '2SLS'} {pooled:.3f}  se {pooled_se:.3f}  95% CI [{pooled - 1.96 * pooled_se:.3f}, {pooled + 1.96 * pooled_se:.3f}]"
          f"   (pooled naive {pooled_naive:.3f})")
    print(f"target ({target_label})   heterogeneity Q {q:.1f} on {len(rows) - 1} df")
    print(f"wrote {sumstats_dir}/*.sumstats.csv, {csv_out}, {png_out}")


if __name__ == "__main__":
    main()
