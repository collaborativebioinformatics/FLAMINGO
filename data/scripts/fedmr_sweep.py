"""Seed sweep: FedMR against the pooled, site-meta and summary-statistic estimators
as one design axis at a time moves away from the default ten-site setting.

Linear causal effect throughout (theta1 = 0.3 unless heterogeneous). Sites are
simulated in memory with the same generators as the checked-in sets, then:

    pooled        concatenated 2SLS (per-site first stages, or one shared first stage
                  when SNPs are shared); FedMR equals it, and the sweep records the
                  largest |FedMR - pooled| seen as the identity check
    fedmr_cf      cross-fitted FedMR, k = 5 (hypothesis: smaller one-sample lean)
    site_meta     each site's own 2SLS, inverse-variance meta-analysis
    sumstats      per-SNP GWAS effects, per-site IVW, meta-analysis

Estimand: what pooled 2SLS with site intercepts converges to when the sites'
causal effects theta_k differ. With Y_k = theta_k X_k + noise, the
coordinator's c = sum_k Z_k'Y_k has expectation sum_k B_k theta_k, so

    theta* = (B' A^-1 sum_k B_k theta_k) / (B' A^-1 B)

a first-stage-weighted mean of the theta_k (for the local-first-stage protocol
the weight of site k is its xhat'X, roughly n_k h2_k), *not* the n_k-weighted
mean. It is computed per replicate from the realised first-stage matrices.
The site meta-analysis and sumstats routes are inverse-variance weighted and
so target a slightly different mean under heterogeneity; their bias is still
reported against theta*. Reported per estimator and level: bias, RMSE, mean
SE, coverage of the 95% interval; also the SNP-set first-stage F.

    uv run python scripts/fedmr_sweep.py --seeds 100          # ~10 min
    uv run python scripts/fedmr_sweep.py --axis sites --seeds 20

Writes results/fedmr_sweep.csv (one row per seed x level x estimator),
results/fedmr_sweep_summary.csv and results/fedmr_sweep.png.
"""

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import numpy as np
import polars as pl

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import flamingo_fedmr as fm

sys.path.insert(0, str(Path(__file__).parent))
from federated_exact_mr import pooled_2sls_shared  # noqa: E402
from federated_nonlinear_mr import sumstats_slope  # noqa: E402
from federated_summary_mr import pooled_2sls  # noqa: E402
from heterogeneity import Heterogeneity  # noqa: E402
from simulate_basic import simulate  # noqa: E402
from simulate_federated_sites import sample_site_params  # noqa: E402

INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e6e5df"
COLORS = {"pooled": "#1f1f1e", "fedmr_cf": "#8a2be2", "site_meta": "#2a78d6", "sumstats": "#eb6834"}
LABELS = {"pooled": "pooled 2SLS = FedMR", "fedmr_cf": "FedMR cross-fitted (5 folds)",
          "site_meta": "site 2SLS meta-analysis", "sumstats": "per-SNP sumstats IVW"}

BASE = dict(n_sites=10, n_snps=20, pop_min=500, pop_max=5000, sizes="spread", h2x_mean=0.10, h2x_kappa=40.0,
            gamma_mean=0.3, gamma_kappa=20.0, theta1=0.3, theta_sd=0.0, shared_snps=False, maf_shift=0.0,
            pleiotropy_mean=0.0, pleiotropy_sd=0.0)

AXES = {
    "sites": [("2", dict(n_sites=2)), ("5", dict(n_sites=5)), ("10", {}), ("20", dict(n_sites=20))],
    "instrument_strength": [("h2 0.02", dict(h2x_mean=0.02)), ("h2 0.05", dict(h2x_mean=0.05)),
                            ("h2 0.10", {}), ("h2 0.20", dict(h2x_mean=0.20))],
    "snps": [("5", dict(n_snps=5)), ("20", {}), ("100", dict(n_snps=100))],
    "imbalance": [("equal", dict(sizes="equal")), ("spread", {}), ("90/10", dict(sizes="90/10"))],
    "allele_frequencies": [("shared, same MAF", dict(shared_snps=True)),
                           ("shared, MAF shift 0.5", dict(shared_snps=True, maf_shift=0.5)),
                           ("shared, MAF shift 1.0", dict(shared_snps=True, maf_shift=1.0))],
    "effect_heterogeneity": [("same", {}), ("theta sd 0.1", dict(theta_sd=0.1)), ("theta sd 0.2", dict(theta_sd=0.2))],
    "pleiotropy": [("none", {}), ("balanced sd 0.02", dict(pleiotropy_sd=0.02)),
                   ("directional mean 0.02", dict(pleiotropy_mean=0.02, pleiotropy_sd=0.02))],
}


def site_sizes(rng: np.random.Generator, cfg: dict) -> np.ndarray | None:
    K = cfg["n_sites"]
    if cfg["sizes"] == "equal":
        return np.full(K, (cfg["pop_min"] + cfg["pop_max"]) // 2)
    if cfg["sizes"] == "90/10":
        total = K * (cfg["pop_min"] + cfg["pop_max"]) // 2
        big = int(0.9 * total)
        rest = np.full(K - 1, (total - big) // max(K - 1, 1))
        return np.array([big, *rest])
    return None   # default spread from sample_site_params


def simulate_sites(cfg: dict, seed: int) -> tuple[list, list]:
    rng = np.random.default_rng(seed)
    n, h2_x, gamma_x, gamma_y = sample_site_params(rng, cfg["n_sites"], cfg["pop_min"], cfg["pop_max"],
                                                    cfg["h2x_mean"], cfg["h2x_kappa"], cfg["gamma_mean"],
                                                    cfg["gamma_kappa"])
    fixed = site_sizes(rng, cfg)
    if fixed is not None:
        n = fixed
    het = Heterogeneity(rng, SimpleNamespace(**cfg))
    sites, thetas = [], []
    for i in range(cfg["n_sites"]):
        site_seed = seed * 1000 + i + 1
        t1 = float(het.theta1[i])
        df, truth = simulate(int(n[i]), cfg["n_snps"], t1, float(h2_x[i]), float(gamma_x[i]), float(gamma_y[i]),
                             site_seed, **het.extra(site_seed))
        G = df.select(pl.col("^snp.*$")).to_numpy().astype(float)
        sites.append(fm.SiteData(f"site{i + 1:02d}", G, df["X"].to_numpy(), df["Y"].to_numpy()))
        thetas.append(t1)
    return sites, thetas


def pooled_estimand(run, thetas: list) -> float:
    """theta* = (B' A^-1 sum_k B_k theta_k) / (B' A^-1 B) from the run's realised per-site
    first-stage matrices: the probability limit of the pooled 2SLS under site-specific effects."""
    st = run.stats
    xi = st.layout.w_index(["X"])[0]
    weighted = np.zeros(len(st.layout.z_names))
    for d, t in zip(run.designs, thetas):
        zr = st.layout.z_index(d.z_names)
        weighted[zr] += (d.Z.T @ d.W[:, xi]) * t
    AinvB = np.linalg.solve(st.A, st.B[:, xi])
    return float(AinvB @ weighted / (AinvB @ st.B[:, xi]))


def estimate_all(sites: list, thetas: list, shared: bool) -> tuple[dict, float, float, float]:
    """Estimates and SEs per route, the identity gap |FedMR - pooled|, the estimand, and the SNP-set F."""
    raw = [(s.G, s.X, s.Y) for s in sites]
    Proto = fm.SharedInstrumentFedMR if shared else fm.LocalFirstStageFedMR
    pooled, pooled_se, _ = pooled_2sls_shared(sites) if shared else pooled_2sls(raw)
    run = Proto(robust=False).run(sites)
    cf = Proto(robust=False, crossfit=5).run(sites).result
    out = {"pooled": (pooled, pooled_se), "fedmr_cf": (cf["X"], cf.se("X"))}
    est, se = zip(*[pooled_2sls([r])[:2] for r in raw])
    w = 1 / np.array(se) ** 2
    out["site_meta"] = (float(np.sum(w * est) / w.sum()), float(np.sqrt(1 / w.sum())))
    out["sumstats"] = sumstats_slope(raw)
    return out, abs(run.result["X"] - pooled), pooled_estimand(run, thetas), run.result.diagnostics.first_stage["X"]["F"]


def run_axis(axis: str, seeds: int) -> list:
    rows = []
    for level, over in AXES[axis]:
        cfg = {**BASE, **over}
        for seed in range(1, seeds + 1):
            sites, thetas = simulate_sites(cfg, seed)
            ests, identity, target, F = estimate_all(sites, thetas, cfg["shared_snps"])
            for name, (e, s) in ests.items():
                rows.append({"axis": axis, "level": level, "seed": seed, "estimator": name, "est": e, "se": s,
                             "target": target, "identity_diff": identity, "first_stage_F": F})
        print(f"{axis:22s} {level:24s} done ({seeds} seeds)", flush=True)
    return rows


def summarize(df: pl.DataFrame) -> pl.DataFrame:
    err = (pl.col("est") - pl.col("target"))
    covered = ((pl.col("est") - 1.96 * pl.col("se") <= pl.col("target")) &
               (pl.col("target") <= pl.col("est") + 1.96 * pl.col("se")))
    return (df.group_by(["axis", "level", "estimator"], maintain_order=True)
              .agg(bias=err.mean(), rmse=(err**2).mean().sqrt(), mean_se=pl.col("se").mean(),
                   coverage=covered.mean(), identity_max=pl.col("identity_diff").max(),
                   mean_F=pl.col("first_stage_F").mean(), seeds=pl.len()))


def plot(summary: pl.DataFrame, out: Path) -> None:
    axes_present = [a for a in AXES if a in summary["axis"].unique().to_list()]
    fig, axs = plt.subplots(2, len(axes_present), figsize=(3.1 * len(axes_present), 6.4), dpi=150, squeeze=False)
    for j, axis in enumerate(axes_present):
        sub = summary.filter(pl.col("axis") == axis)
        levels = [lv for lv, _ in AXES[axis]]
        x = np.arange(len(levels))
        for i, est in enumerate(COLORS):
            s = sub.filter(pl.col("estimator") == est)
            by = {lv: (b, r, c) for lv, b, r, c in zip(s["level"], s["bias"], s["rmse"], s["coverage"])}
            b = [by.get(lv, (np.nan,) * 3)[0] for lv in levels]
            r = [by.get(lv, (np.nan,) * 3)[1] for lv in levels]
            c = [by.get(lv, (np.nan,) * 3)[2] for lv in levels]
            off = (i - 1.5) * 0.12
            axs[0, j].errorbar(x + off, b, yerr=r, fmt="o", ms=4, color=COLORS[est], capsize=2, label=LABELS[est])
            axs[1, j].plot(x + off, c, "o", ms=4, color=COLORS[est])
        axs[0, j].axhline(0, color=INK, linewidth=0.8, linestyle="--")
        axs[1, j].axhline(0.95, color=INK, linewidth=0.8, linestyle="--")
        axs[0, j].set_title(axis.replace("_", " "), loc="left", fontsize=10, color=INK)
        for r_ in (0, 1):
            axs[r_, j].set_xticks(x); axs[r_, j].set_xticklabels(levels, rotation=25, ha="right", fontsize=8)
            for sp in ("top", "right"):
                axs[r_, j].spines[sp].set_visible(False)
            axs[r_, j].grid(color=GRID, linewidth=0.6); axs[r_, j].set_axisbelow(True)
        axs[1, j].set_ylim(0.5, 1.0)
    axs[0, 0].set_ylabel("bias (dot) and RMSE (bar)")
    axs[1, 0].set_ylabel("95% coverage")
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", ncol=4, fontsize=9)
    fig.suptitle("FedMR seed sweep: estimand = first-stage-weighted mean site effect (the pooled 2SLS limit)",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seeds", type=int, default=100)
    p.add_argument("--axis", action="append", choices=list(AXES), help="repeatable; default all")
    p.add_argument("--out", type=Path, default=Path("results/fedmr_sweep"))
    a = p.parse_args()
    rows = []
    for axis in a.axis or list(AXES):
        rows += run_axis(axis, a.seeds)
    df = pl.DataFrame(rows)
    summary = summarize(df)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(a.out.with_suffix(".csv"))
    summary.write_csv(Path(f"{a.out}_summary.csv"))
    plot(summary, a.out.with_suffix(".png"))
    with pl.Config(tbl_rows=-1, tbl_cols=-1, float_precision=4, tbl_width_chars=160):
        print(summary)
    print(f"largest |FedMR - pooled| over the whole sweep: {df['identity_diff'].max():.2e}")
    print(f"wrote {a.out}.csv, {a.out}_summary.csv, {a.out}.png")


if __name__ == "__main__":
    main()
