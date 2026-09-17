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

SITE_COLOR, SUMSTATS_COLOR, FED_COLOR, FEDMR_COLOR, INK, MUTED, GRID = ("#2a78d6", "#eb6834", "#1baf7a", "#8a2be2",
                                                                     "#1f1f1e", "#6b6a63", "#e6e5df")
FL_RESULTS = Path(__file__).resolve().parents[2] / "federated_learning" / "results"


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


def tsls(D, Z, Y):
    """Generic 2SLS of Y on regressors D with instruments Z. Returns (coef, covariance)."""
    Dhat = Z @ np.linalg.lstsq(Z, D, rcond=None)[0]
    coef = np.linalg.lstsq(Dhat, Y, rcond=None)[0]
    resid = Y - D @ coef
    sigma2 = resid @ resid / (len(Y) - D.shape[1])
    return coef, sigma2 * np.linalg.inv(Dhat.T @ Dhat)


def quadratic_2sls(sites):
    """2SLS with X and X^2 as endogenous regressors, instrumented by the SNP-predicted X and its
    square, with site intercepts. Returns (theta[2], cov[2x2]). A single site is a list of one."""
    xhat, X, Y, site_idx = _first_stage(sites)
    S = np.eye(len(sites))[site_idx]
    coef, cov = tsls(np.column_stack([X, X**2, S]), np.column_stack([xhat, xhat**2, S]), Y)
    return coef[:2], cov[:2, :2]


def multivariate_meta(thetas, covs):
    """Inverse-variance meta-analysis of vector estimates with their covariances."""
    W = [np.linalg.inv(c) for c in covs]
    cov = np.linalg.inv(sum(W))
    theta = cov @ sum(w @ t for w, t in zip(W, thetas))
    return theta, cov


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


def site_summary(csv: Path, sumstats_dir: Path, survival: bool, curved: bool = False) -> dict:
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
    row = {"site": site, "n": df.height, "avg_slope": truth["avg_slope"],
           "ivw": est, "ivw_se": se, "naive_ols": naive, "mean_F": f_stat}
    if curved:
        # site-level model summary: the site fits the quadratic 2SLS on its own data and shares
        # the two coefficients and their covariance, never individual rows
        th, cov = quadratic_2sls([site_summary.raw[-1]])
        site_summary.models.append((th, cov))
        row.update({"q_theta1": float(th[0]), "q_theta1_se": float(np.sqrt(cov[0, 0])),
                    "q_theta2": float(th[1]), "q_theta2_se": float(np.sqrt(cov[1, 1]))})
    return row


site_summary.raw = []
site_summary.models = []


def _family_rows(ax, y_sites, rows_y, res, n_all, families):
    """Y tick labels for site rows and the family-grouped combined rows."""
    ticks = list(y_sites) + [rows_y[k] for k in families]
    labels = [f"{s}  (n={n:,}, F={f:.0f})" for s, n, f in zip(res["site"], res["n"], res["mean_F"])]
    labels += [f"{families[k]}  (n={n_all:,})" for k in families]
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=9)


def forest(res, meta, meta_se, pooled, pooled_se, pooled_naive, fl, target, target_label, shape, out: Path,
           fedmr=None):
    sites = res["site"].to_list()
    y = np.arange(len(sites))[::-1]
    rows_y = {"sumstats": -1, "federated": -2.2, "fedmr": -3.4, "pooled": -4.6}
    fig, ax = plt.subplots(figsize=(8, 7.2), dpi=150)
    ax.axvline(target, color=INK, linewidth=1.2, linestyle="--")
    ax.errorbar(res["ivw"], y, xerr=1.96 * res["ivw_se"], fmt="o", color=SITE_COLOR, ms=6,
                ecolor=SITE_COLOR, elinewidth=2, capsize=0, label="site IVW (95% CI)")
    ax.scatter(res["naive_ols"], y, marker="|", s=120, color=MUTED, linewidths=2, label="naive fit (no instruments)", zorder=3)
    ax.errorbar([meta], [rows_y["sumstats"]], xerr=[1.96 * meta_se], fmt="D", color=SUMSTATS_COLOR, ms=8,
                ecolor=SUMSTATS_COLOR, elinewidth=3, label="sumstats: meta-analysis of site IVW")
    if "2sri" in fl:
        ax.scatter([fl["2sri"][0]], [rows_y["federated"]], marker="^", s=90, color=FED_COLOR, zorder=4,
                   label="federated: NVFlare 2SRI, global model (no CI)")
    if "naive" in fl:
        ax.scatter([fl["naive"][0]], [rows_y["federated"]], marker="|", s=120, color=MUTED, linewidths=2, zorder=3)
    if fedmr is not None:
        ax.errorbar([fedmr[0]], [rows_y["fedmr"]], xerr=[1.96 * fedmr[1]], fmt="v", color=FEDMR_COLOR, ms=8,
                    ecolor=FEDMR_COLOR, elinewidth=3, label="federated: FedMR, exact pooled 2SLS (95% CI)")
    else:
        ax.text(0.5, rows_y["fedmr"], "no FedMR run for this dataset", transform=ax.get_yaxis_transform(),
                ha="center", va="center", fontsize=9, color=FEDMR_COLOR, style="italic")
    pooled_label = "concatenated: one stratified 2SPS Cox" if shape == "cox" else "concatenated: one pooled 2SLS"
    ax.errorbar([pooled], [rows_y["pooled"]], xerr=[1.96 * pooled_se], fmt="s", color=INK, ms=7,
                ecolor=INK, elinewidth=3, label=pooled_label)
    ax.scatter([pooled_naive], [rows_y["pooled"]], marker="|", s=120, color=MUTED, linewidths=2, zorder=3)
    ax.axhline(-0.4, color=GRID, linewidth=0.8)
    if "2sri" not in fl:
        ax.text(0.5, rows_y["federated"], "no NVFlare 2SRI run for this dataset", transform=ax.get_yaxis_transform(),
                ha="center", va="center", fontsize=9, color=FED_COLOR, style="italic")
    _family_rows(ax, y, rows_y, res, res["n"].sum(),
                 {"sumstats": "sumstats", "federated": "federated: FedAvg", "fedmr": "federated: FedMR",
                  "pooled": "concatenated"})
    ax.set_xlabel("estimated log hazard ratio per unit X" if shape == "cox" else "estimated causal effect of X on Y")
    ax.set_title(f"{shape} model: sumstats vs federated vs concatenated\ndashed line: {target_label}",
                 loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=9)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out)


def fedmr_row(dataset: str, curved: bool):
    """FedMR estimate from results/fedmr.<dataset>.csv (scripts/federated_exact_mr.py), or None.
    Returns (theta1, se1, theta2, se2); theta2 and se2 are None for the linear layout."""
    path = Path("results") / f"fedmr.{dataset}.csv"
    if not path.exists():
        return None
    df = pl.read_csv(path)
    r = df.filter(pl.col("estimator") == "FedMR quadratic" if curved else pl.col("estimator").str.starts_with("FedMR ("))
    if r.height == 0:
        return None
    r = r.row(0, named=True)
    return r["theta1"], r["se1"], r["theta2"], r["se2"]


def fl_curve(method: str, dataset: str):
    """Last-round global-model curve f(x) from the NVFlare run, or None if that run is missing."""
    path = FL_RESULTS / method / dataset / "curves.csv"
    if not path.exists():
        return None
    df = pl.read_csv(path)
    last = df.filter(pl.col("round") == pl.col("round").max())
    c = last.group_by("x").agg(pl.col("f").mean()).sort("x")
    return c["x"].to_numpy(), c["f"].to_numpy()


def fl_params(dataset: str, curved: bool, x_max: float = 2.0):
    """Summarise the federated curves into the forest's parameters by least squares on |x| <= x_max.
    Returns {"2sri": (theta1, theta2) or (slope,), "naive": ...} for the runs that exist."""
    out = {}
    for method in ("2sri", "naive"):
        c = fl_curve(method, dataset)
        if c is None:
            continue
        x, f = c
        keep = np.abs(x) <= x_max
        basis = np.column_stack([x[keep], x[keep] ** 2]) if curved else x[keep][:, None]
        coef = np.linalg.lstsq(basis, f[keep] - np.interp(0.0, x, f), rcond=None)[0]
        out[method] = tuple(float(v) for v in coef)
    return out


def _style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=MUTED)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def forest_curved(res, meta, meta_se, model_meta, model_cov, pooled_q, pooled_cov, pooled_naive, fl,
                  avg_slope_target, shape, theta1, theta2, out: Path, fedmr=None):
    """Two columns, one per parameter of the quadratic basis. Rows: sites, then the three families."""
    sites = res["site"].to_list()
    y = np.arange(len(sites))[::-1]
    rows_y = {"sumstats": -1, "models": -1.9, "federated": -3.1, "fedmr": -4.3, "pooled": -5.5}
    fig, axes = plt.subplots(1, 2, figsize=(11, 7.4), dpi=150, sharey=True,
                             gridspec_kw={"width_ratios": [1.15, 1]})
    ax1, ax2 = axes
    n_all = res["n"].sum()

    # column 1: theta1 (slope at X = 0)
    ax1.axvline(theta1, color=INK, linewidth=1.2, linestyle="--")
    if abs(avg_slope_target - theta1) > 0.01:
        ax1.axvline(avg_slope_target, color=MUTED, linewidth=1.2, linestyle=":")
    ax1.errorbar(res["q_theta1"], y, xerr=1.96 * res["q_theta1_se"], fmt="o", color=SITE_COLOR, ms=6,
                 ecolor=SITE_COLOR, elinewidth=2, label="site model: local quadratic 2SLS (95% CI)")
    ax1.scatter(res["ivw"], y, marker="o", s=34, facecolor="white", edgecolor=SUMSTATS_COLOR, linewidths=1.6, zorder=4,
                label="site per-SNP IVW (average slope)")
    ax1.scatter(res["naive_ols"], y, marker="|", s=120, color=MUTED, linewidths=2, zorder=3, label="naive fit (no instruments)")
    ax1.errorbar([meta], [rows_y["sumstats"]], xerr=[1.96 * meta_se], fmt="D", color=SUMSTATS_COLOR, ms=8,
                 ecolor=SUMSTATS_COLOR, elinewidth=3, label="per-SNP sumstats: meta of site IVW")
    ax1.errorbar([model_meta[0]], [rows_y["models"]], xerr=[1.96 * np.sqrt(model_cov[0, 0])], fmt="D",
                 mfc="white", color=SUMSTATS_COLOR, ms=8, ecolor=SUMSTATS_COLOR, elinewidth=3,
                 label="model sumstats: meta of site (θ1, θ2)")
    if "2sri" in fl:
        ax1.scatter([fl["2sri"][0]], [rows_y["federated"]], marker="^", s=90, color=FED_COLOR, zorder=4,
                    label="federated: NVFlare 2SRI, global model (no CI)")
        ax2.scatter([fl["2sri"][1]], [rows_y["federated"]], marker="^", s=90, color=FED_COLOR, zorder=4)
    else:
        ax1.text(0.5, rows_y["federated"], "no NVFlare 2SRI run for this dataset", transform=ax1.get_yaxis_transform(),
                 ha="center", va="center", fontsize=9, color=FED_COLOR, style="italic")
    if "naive" in fl:
        ax1.scatter([fl["naive"][0]], [rows_y["federated"]], marker="|", s=120, color=MUTED, linewidths=2, zorder=3)
        ax2.scatter([fl["naive"][1]], [rows_y["federated"]], marker="|", s=120, color=MUTED, linewidths=2, zorder=3)
    if fedmr is not None:
        ax1.errorbar([fedmr[0]], [rows_y["fedmr"]], xerr=[1.96 * fedmr[1]], fmt="v", color=FEDMR_COLOR, ms=8,
                     ecolor=FEDMR_COLOR, elinewidth=3, label="federated: FedMR, exact pooled 2SLS (95% CI)")
        ax2.errorbar([fedmr[2]], [rows_y["fedmr"]], xerr=[1.96 * fedmr[3]], fmt="v", color=FEDMR_COLOR, ms=8,
                     ecolor=FEDMR_COLOR, elinewidth=3)
    else:
        ax1.text(0.5, rows_y["fedmr"], "no FedMR run for this dataset", transform=ax1.get_yaxis_transform(),
                 ha="center", va="center", fontsize=9, color=FEDMR_COLOR, style="italic")
    ax1.errorbar([pooled_q[0]], [rows_y["pooled"]], xerr=[1.96 * np.sqrt(pooled_cov[0, 0])], fmt="s", color=INK,
                 ms=7, ecolor=INK, elinewidth=3, label="concatenated: one quadratic 2SLS")
    ax1.scatter([pooled_naive], [rows_y["pooled"]], marker="|", s=120, color=MUTED, linewidths=2, zorder=3)
    ax1.set_xlabel("θ1: slope at X = 0")
    slope_note = "" if abs(avg_slope_target - theta1) <= 0.01 else f";  dotted: average slope {avg_slope_target:.2f}"
    ax1.set_title(f"dashed: true θ1 = {theta1}{slope_note}", loc="left", fontsize=10, color=INK)

    # column 2: theta2 (curvature)
    if shape == "quadratic":
        ax2.axvline(theta2, color=INK, linewidth=1.2, linestyle="--")
        ax2.set_title(f"dashed: true θ2 = {theta2}", loc="left", fontsize=10, color=INK)
    else:
        ax2.axvline(0, color=GRID, linewidth=1.2)
        ax2.set_title(f"no true θ2 (kink at X = {theta2}); negative = saturation",
                      loc="left", fontsize=10, color=INK)
    ax2.errorbar(res["q_theta2"], y, xerr=1.96 * res["q_theta2_se"], fmt="o", color=SITE_COLOR, ms=6,
                 ecolor=SITE_COLOR, elinewidth=2)
    ax2.errorbar([model_meta[1]], [rows_y["models"]], xerr=[1.96 * np.sqrt(model_cov[1, 1])], fmt="D",
                 mfc="white", color=SUMSTATS_COLOR, ms=8, ecolor=SUMSTATS_COLOR, elinewidth=3)
    ax2.errorbar([pooled_q[1]], [rows_y["pooled"]], xerr=[1.96 * np.sqrt(pooled_cov[1, 1])], fmt="s", color=INK,
                 ms=7, ecolor=INK, elinewidth=3)
    ax2.text(0.5, rows_y["sumstats"], "not identifiable from per-SNP summary statistics", transform=ax2.get_yaxis_transform(),
             ha="center", va="center", fontsize=9, color=SUMSTATS_COLOR, style="italic")
    ax2.set_xlabel("θ2: curvature")

    for ax in axes:
        ax.axhline(-0.4, color=GRID, linewidth=0.8)
        _style(ax)
    _family_rows(ax1, y, rows_y, res, n_all, {"sumstats": "sumstats: per-SNP", "models": "sumstats: site models",
                                              "federated": "federated: FedAvg", "fedmr": "federated: FedMR",
                                              "pooled": "concatenated"})
    fig.suptitle(f"{shape} model: sumstats vs federated vs concatenated, two parameters of the causal curve",
                 x=0.01, ha="left", fontsize=11, color=INK)
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
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
    site_summary.raw, site_summary.models = [], []
    curved = a.shape in ("quadratic", "threshold")
    rows = [site_summary(csv, sumstats_dir, a.shape == "cox", curved) for csv in sorted(a.sites.glob("site*.csv"))]
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
    fl = fl_params(a.sites.name, curved)
    fedmr = fedmr_row(a.sites.name, curved)
    if fedmr is None:
        print(f"no FedMR result at results/fedmr.{a.sites.name}.csv; run scripts/federated_exact_mr.py --shape {a.shape}")
    else:
        print(f"federated FedMR: " + "  ".join(f"{v:.3f}" for v in fedmr if v is not None) + "   (exact, with CI)")
    for method, coef in fl.items():
        print(f"federated NVFlare {method:5s}: " + "  ".join(f"{v:.3f}" for v in coef) + "   (from curve, no CI)")
    if "2sri" not in fl:
        print(f"no NVFlare 2SRI run at {FL_RESULTS / '2sri' / a.sites.name}")
    if curved:
        model_meta, model_cov = multivariate_meta(*zip(*site_summary.models))
        pooled_q, pooled_cov = quadratic_2sls(site_summary.raw)
        forest_curved(res, meta, meta_se, model_meta, model_cov, pooled_q, pooled_cov, pooled_naive, fl,
                      target, a.shape, manifest["theta1"], manifest["theta2"], png_out, fedmr)
        mse = np.sqrt(np.diag(model_cov)); pse = np.sqrt(np.diag(pooled_cov))
        print(f"model sumstats (meta of site quadratic fits): theta1 {model_meta[0]:.3f} ({mse[0]:.3f})  "
              f"theta2 {model_meta[1]:.3f} ({mse[1]:.3f})")
        print(f"concatenated quadratic 2SLS:                  theta1 {pooled_q[0]:.3f} ({pse[0]:.3f})  "
              f"theta2 {pooled_q[1]:.3f} ({pse[1]:.3f})")
    else:
        forest(res, meta, meta_se, pooled, pooled_se, pooled_naive, fl, target, target_label, a.shape, png_out, fedmr)

    with pl.Config(tbl_rows=-1, float_precision=3):
        print(res)
    print(f"\nmeta IVW  {meta:.3f}  se {meta_se:.3f}  95% CI [{meta - 1.96 * meta_se:.3f}, {meta + 1.96 * meta_se:.3f}]")
    print(f"pooled {'2SPS Cox' if a.shape == 'cox' else '2SLS'} {pooled:.3f}  se {pooled_se:.3f}  95% CI [{pooled - 1.96 * pooled_se:.3f}, {pooled + 1.96 * pooled_se:.3f}]"
          f"   (pooled naive {pooled_naive:.3f})")
    print(f"target ({target_label})   heterogeneity Q {q:.1f} on {len(rows) - 1} df")
    print(f"wrote {sumstats_dir}/*.sumstats.csv, {csv_out}, {png_out}")


if __name__ == "__main__":
    main()
