"""Plot per-client test metrics across federated rounds.

    uv run python plot_metrics.py                      # reads workspace/metrics/*.csv
    uv run python plot_metrics.py --stage local        # metrics after local epochs
    uv run python plot_metrics.py --out results/x.png

One panel per metric. Thin gray lines are the ten sites (labeled at the right
edge), the bold blue line is the test-size-weighted mean across sites.
"""

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
METRICS = ["accuracy", "precision", "recall", "f1", "auc", "loss"]
BLUE, GRAY, INK, MUTED, SURFACE = "#256abf", "#b8b8b5", "#1f1f1e", "#6b6b68", "#fcfcfb"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metrics_dir", default=os.path.join(HERE, "workspace", "metrics"))
    p.add_argument("--stage", default="global", choices=["global", "local"])
    p.add_argument("--out", default=None)
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.metrics_dir, "*.csv")))
    if not files:
        raise SystemExit(f"no metrics in {args.metrics_dir}; run job.py first")
    df = pd.concat(pd.read_csv(f) for f in files)
    df = df[df.stage == args.stage]
    rounds = sorted(df["round"].unique())
    sites = sorted(df.site.unique())

    wmean = (df.groupby("round")
               .apply(lambda g: pd.Series({m: (g[m] * g.n_test).sum() / g.n_test.sum() for m in METRICS}),
                      include_groups=False))

    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), facecolor=SURFACE)
    for ax, m in zip(axes.flat, METRICS):
        ax.set_facecolor(SURFACE)
        last = df[df["round"] == rounds[-1]].set_index("site")[m]
        # Label sites only where they spread apart; stacked labels are unreadable.
        label_sites = (last.max() - last.min()) > 0.15 * (df[m].max() - df[m].min())
        for s in sites:
            g = df[df.site == s].sort_values("round")
            ax.plot(g["round"], g[m], color=GRAY, lw=1.2, zorder=1)
            if label_sites:
                ax.annotate(s[-2:], (rounds[-1], last[s]), xytext=(3, 0),
                            textcoords="offset points", fontsize=6.5, color=MUTED, va="center")
        ax.plot(wmean.index, wmean[m], color=BLUE, lw=2.2, marker="o", ms=4.5, zorder=3)
        ax.annotate(f"{wmean[m].iloc[-1]:.3f}", (rounds[-1], wmean[m].iloc[-1]), xytext=(-6, 9),
                    textcoords="offset points", fontsize=8, color=INK, fontweight="bold", ha="right")
        ax.set_title(m.upper() if m == "auc" else m.capitalize(), loc="left", fontsize=10, color=INK)
        ax.set_xticks(rounds)
        ax.grid(axis="y", color="#e6e6e3", lw=0.8)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#cfcfcc")
        ax.tick_params(colors=MUTED, labelsize=8, length=0)
        ax.set_xlim(rounds[0] - 0.2, rounds[-1] + 0.8)
    for ax in axes[1]:
        ax.set_xlabel("federated round", fontsize=8, color=MUTED)

    label = "global model as received" if args.stage == "global" else "model after local epochs"
    fig.suptitle(f"Per-site test metrics per round ({label})", x=0.01, ha="left",
                 fontsize=12, color=INK, fontweight="bold")
    fig.text(0.01, 0.935, "Gray: each of the ten sites on its own 20% test split.  "
             "Blue: test-size-weighted mean across sites.", fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    out = args.out or os.path.join(HERE, "results", f"metrics_by_round.{args.stage}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
