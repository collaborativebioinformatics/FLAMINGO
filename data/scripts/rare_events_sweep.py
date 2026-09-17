"""Scenario 2: few events per site. Repeat the ten-site Cox draw over seeds and
score the summary-statistic route against the concatenated route.

Per seed: draw ten sites (small populations, short follow-up), run per-SNP Cox
GWAS at each site, IVW per site, inverse-variance meta-analysis; and one 2SPS
Cox on the concatenated data stratified by site. Records estimate, standard
error, and whether the 95% CI covers the true log hazard ratio.

Writes results/rare_events.csv (one row per seed) and results/rare_events.png.
"""

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import polars as pl

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from federated_summary_mr import cox_gwas, gwas, ivw, pooled_2sps_cox  # noqa: E402
from simulate_basic import simulate_survival  # noqa: E402
from simulate_federated_sites import sample_site_params  # noqa: E402

SUMSTATS_COLOR, POOLED_COLOR, INK, MUTED, GRID = "#eb6834", "#2a78d6", "#1f1f1e", "#6b6a63", "#e6e5df"


def one_draw(seed, a):
    rng = np.random.default_rng(seed)
    n, h2_x, gamma_x, gamma_y = sample_site_params(rng, a.n_sites, a.pop_min, a.pop_max, 0.10, 40.0, 0.3, 20.0)
    sites, est, w, events = [], [], [], []
    for i in range(a.n_sites):
        df, _ = simulate_survival(int(n[i]), a.n_snps, a.theta, float(h2_x[i]), float(gamma_x[i]),
                                  float(gamma_y[i]), seed * 100 + i, censor_frac=a.censor_frac, followup=a.followup)
        G = df.select(pl.col("^snp.*$")).to_numpy().astype(float)
        X, time, event = df["X"].to_numpy(), df["time"].to_numpy(), df["event"].to_numpy()
        sites.append((G, X, np.column_stack([time, event])))
        events.append(int(event.sum()))
        bx, _ = gwas(G, X)
        by, se_y = cox_gwas(G, time, event)
        e, s = ivw(bx, by, se_y)
        est.append(e); w.append(1 / s**2)
    est, w = np.array(est), np.array(w)
    meta, meta_se = float(np.sum(w * est) / np.sum(w)), float(np.sqrt(1 / np.sum(w)))
    q = float(np.sum(w * (est - meta) ** 2))
    pooled, pooled_se, _ = pooled_2sps_cox(sites)
    return {"seed": seed, "min_events": min(events), "median_events": float(np.median(events)),
            "sumstats": meta, "sumstats_se": meta_se, "Q": q, "pooled": pooled, "pooled_se": pooled_se}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seeds", type=int, default=30)
    p.add_argument("--n-sites", type=int, default=10)
    p.add_argument("--n-snps", type=int, default=20)
    p.add_argument("--pop-min", type=int, default=300)
    p.add_argument("--pop-max", type=int, default=3000)
    p.add_argument("--theta", type=float, default=0.3)
    p.add_argument("--censor-frac", type=float, default=0.0)
    p.add_argument("--followup", type=float, default=1.2)
    p.add_argument("--out", type=Path, default=Path("results/rare_events"))
    a = p.parse_args()

    warnings.filterwarnings("ignore")
    rows = []
    for seed in range(1, a.seeds + 1):
        rows.append(one_draw(seed, a))
        r = rows[-1]
        print(f"seed {seed:3d}  events min {r['min_events']:3d} median {r['median_events']:5.0f}  "
              f"sumstats {r['sumstats']:.3f} ({r['sumstats_se']:.3f})  pooled {r['pooled']:.3f} ({r['pooled_se']:.3f})  Q {r['Q']:.1f}",
              flush=True)
    res = pl.DataFrame(rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    res.write_csv(a.out.with_name(a.out.name + ".csv"))

    print(f"\n{a.seeds} seeds, true log HR {a.theta}, events per site: median of medians "
          f"{res['median_events'].median():.0f}, smallest site {res['min_events'].min()}")
    print(f"{'route':9s} {'mean':>7s} {'bias':>7s} {'emp sd':>7s} {'mean se':>8s} {'RMSE':>7s} {'95% cov':>8s}")
    summary = {}
    for route in ("sumstats", "pooled"):
        e, s = res[route].to_numpy(), res[f"{route}_se"].to_numpy()
        cover = np.mean(np.abs(e - a.theta) <= 1.96 * s)
        summary[route] = (e, s, cover)
        print(f"{route:9s} {e.mean():7.3f} {e.mean() - a.theta:+7.3f} {e.std(ddof=1):7.3f} {s.mean():8.3f} "
              f"{np.sqrt(np.mean((e - a.theta) ** 2)):7.3f} {cover:8.2f}")
    print(f"heterogeneity Q: mean {res['Q'].mean():.1f} on {a.n_sites - 1} df  (expected {a.n_sites - 1} if SEs are right)")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=150, sharey=True)
    for ax, (route, color, label) in zip(axes, (("sumstats", SUMSTATS_COLOR, "sumstats: meta of site IVW"),
                                                ("pooled", POOLED_COLOR, "concatenated: stratified 2SPS Cox"))):
        e, s, cover = summary[route]
        order = np.argsort(e)
        y = np.arange(len(e))
        covered = np.abs(e - a.theta) <= 1.96 * s
        for yi, i in zip(y, order):
            ax.plot([e[i] - 1.96 * s[i], e[i] + 1.96 * s[i]], [yi, yi], color=color if covered[i] else MUTED,
                    linewidth=1.5, alpha=0.9)
        ax.scatter(e[order], y, color=color, s=12, zorder=3)
        ax.axvline(a.theta, color=INK, linestyle="--", linewidth=1.2)
        ax.set_title(f"{label}\n95% CI coverage {cover:.0%}, mean se {s.mean():.3f}, empirical sd {e.std(ddof=1):.3f}",
                     loc="left", fontsize=10, color=INK)
        ax.set_xlabel("estimated log hazard ratio (dashed: truth)")
        ax.set_yticks([])
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=MUTED)
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel(f"{a.seeds} seeds, sorted by estimate")
    fig.suptitle(f"Few events per site: {a.n_sites} sites of {a.pop_min:,}-{a.pop_max:,} people, follow-up {a.followup}",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(a.out.with_name(a.out.name + ".png"))
    print(f"wrote {a.out}.csv and {a.out}.png")


if __name__ == "__main__":
    main()
