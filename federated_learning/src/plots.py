"""Plots for one dataset's federated run, from results/<dataset>/{metrics,curves}.csv.

    uv run python src/plots.py                    # re-render every dataset + overview
    uv run python src/plots.py quadratic          # one dataset

metrics_by_round.{global,local}.png  one panel per metric across rounds; thin
    gray lines are the sites, bold blue the test-size-weighted mean.
fitted_curve.png  the global model's f(X) at the last round against the true
    causal curve from manifest.json and the pooled empirical E[outcome | X].
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tasks import X_GRID, detect_task, load_manifest, true_curve  # noqa: E402

BLUE, ORANGE, GRAY, INK, MUTED, SURFACE, GRID = (
    "#256abf", "#eb6834", "#b8b8b5", "#1f1f1e", "#6b6b68", "#fcfcfb", "#e6e6e3")


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, lw=0.8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#cfcfcc")
    ax.tick_params(colors=MUTED, labelsize=8, length=0)


def plot_metrics(df, task, stage, out, dataset):
    df = df[df.stage == stage]
    rounds = sorted(df["round"].unique())
    sites = sorted(df.site.unique())
    metrics = [m for m in task.metrics if m != "events"]
    wmean = df.groupby("round").apply(
        lambda g: pd.Series({m: (g[m] * g.n_test).sum() / g.n_test.sum() for m in metrics}),
        include_groups=False)

    ncol = 3 if len(metrics) > 4 else len(metrics)
    nrow = int(np.ceil(len(metrics) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.2 * nrow + 0.6), facecolor=SURFACE, squeeze=False)
    for ax in axes.flat[len(metrics):]:
        ax.axis("off")
    for ax, m in zip(axes.flat, metrics):
        _style(ax)
        last = df[df["round"] == rounds[-1]].set_index("site")[m]
        label_sites = (last.max() - last.min()) > 0.15 * (df[m].max() - df[m].min() + 1e-12)
        for s in sites:
            g = df[df.site == s].sort_values("round")
            ax.plot(g["round"], g[m], color=GRAY, lw=1.2, zorder=1)
            if label_sites:
                ax.annotate(s[-2:], (rounds[-1], last[s]), xytext=(3, 0), textcoords="offset points",
                            fontsize=6.5, color=MUTED, va="center")
        ax.plot(wmean.index, wmean[m], color=BLUE, lw=2.2, marker="o", ms=4.5, zorder=3)
        ax.annotate(f"{wmean[m].iloc[-1]:.3f}", (rounds[-1], wmean[m].iloc[-1]), xytext=(-6, 9),
                    textcoords="offset points", fontsize=8, color=INK, fontweight="bold", ha="right")
        ax.set_title(m.upper() if m in ("auc", "mse", "mae", "r2") else m.replace("_", " ").capitalize(),
                     loc="left", fontsize=10, color=INK)
        ax.set_xticks(rounds)
        ax.set_xlim(rounds[0] - 0.2, rounds[-1] + 0.8)
    for ax in axes[-1]:
        ax.set_xlabel("federated round", fontsize=8, color=MUTED)
    label = "global model as received" if stage == "global" else "model after local epochs"
    fig.suptitle(f"{dataset}: per-site test metrics per round ({label})", x=0.01, ha="left",
                 fontsize=12, color=INK, fontweight="bold")
    fig.text(0.01, 0.93 if nrow > 1 else 0.88,
             "Gray: each site on its own 20% test split.  Blue: test-size-weighted mean across sites.",
             fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.92 if nrow > 1 else 0.86))
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def draw_curve(ax, curves, task, manifest, data_dir, title):
    last = curves["round"].max()
    c = curves[(curves["round"] == last)].groupby("x")["f"].mean().sort_index()
    x, f = c.index.to_numpy(), c.to_numpy()
    f = getattr(task, "curve_transform", lambda v: v)(f)
    _style(ax)
    ax.grid(axis="x", color=GRID, lw=0.8)

    # Pooled empirical E[outcome | X] in bins: the association the model is fitting.
    if task.name != "survival":
        pooled = pd.concat(pd.read_csv(os.path.join(data_dir, f)) for f in sorted(os.listdir(data_dir))
                           if f.startswith("site") and f.endswith(".csv"))
        pooled["bin"] = pd.cut(pooled.X, np.linspace(-3, 3, 25))
        emp = pooled.groupby("bin", observed=True).agg(x=("X", "mean"), y=("Y", "mean"), n=("Y", "size"))
        emp = emp[emp.n >= 30]
        ax.scatter(emp.x, emp.y, s=18, color=GRAY, zorder=2, label="pooled data, binned mean of outcome")

    tc = true_curve(manifest, x)
    if tc is not None:
        ax.plot(x, tc, color=ORANGE, lw=2, ls="--", zorder=3, label="true causal curve f(X)")
    ax.plot(x, f, color=BLUE, lw=2.4, zorder=4, label=f"federated model, round {last}")
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.set_xlabel("X (exposure)", fontsize=9, color=MUTED)
    ax.set_ylabel(task.curve_label, fontsize=9, color=MUTED)


NOTE = ("The model fits E[outcome | X], which includes the confounder U's path, "
        "so it need not match the causal curve.")


def plot_curve(curves, task, manifest, data_dir, out, dataset):
    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=SURFACE)
    draw_curve(ax, curves, task, manifest, data_dir, "")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.suptitle(f"{dataset}: fitted X -> outcome curve", x=0.01, ha="left", fontsize=12, color=INK, fontweight="bold")
    fig.text(0.01, 0.9, NOTE, fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_overview(datasets, results_root, fed_dir, out):
    """One panel per dataset: fitted curve vs truth, plus a summary CSV of last-round metrics."""
    ncol = 3
    nrow = int(np.ceil(len(datasets) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.8 * nrow + 0.8), facecolor=SURFACE, squeeze=False)
    for ax in axes.flat[len(datasets):]:
        ax.axis("off")
    rows = []
    for ax, ds in zip(axes.flat, datasets):
        data_dir = os.path.join(fed_dir, ds)
        manifest = load_manifest(data_dir)
        task = detect_task(pd.read_csv(os.path.join(data_dir, "site01.csv"), nrows=2000), manifest)
        curves = pd.read_csv(os.path.join(results_root, ds, "curves.csv"))
        metrics = pd.read_csv(os.path.join(results_root, ds, "metrics.csv"))
        g = metrics[(metrics["round"] == metrics["round"].max()) & (metrics.stage == "global")]
        w = g.n_test
        score = {m: (g[m] * w).sum() / w.sum() for m in task.metrics if m not in ("loss", "events")}
        rows.append({"dataset": ds, "task": task.name, "round": int(g["round"].iloc[0]), **score})
        draw_curve(ax, curves, task, manifest, data_dir,
                   f"{ds} ({task.name})\n" + ", ".join(f"{k} {v:.2f}" for k, v in list(score.items())[:2]))
    handles, labels = {}, []
    for ax in axes.flat:
        for h, l in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(l, h)
    fig.legend(handles.values(), handles.keys(), frameon=False, fontsize=8.5, loc="upper right", ncol=3,
               bbox_to_anchor=(0.99, 0.985))
    fig.suptitle("Federated fit of X -> outcome under every simulated outcome model", x=0.01, ha="left",
                 fontsize=12, color=INK, fontweight="bold")
    fig.text(0.01, 0.935, NOTE + "  Titles show the last-round weighted test metrics.", fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(results_root, "summary.csv"), index=False)
    return summary


def plot_dataset(dataset, results_dir, data_dir, task=None):
    metrics = pd.read_csv(os.path.join(results_dir, "metrics.csv"))
    curves = pd.read_csv(os.path.join(results_dir, "curves.csv"))
    manifest = load_manifest(data_dir)
    if task is None:
        task = detect_task(pd.read_csv(os.path.join(data_dir, "site01.csv"), nrows=2000), manifest)
    for stage in ("global", "local"):
        plot_metrics(metrics, task, stage, os.path.join(results_dir, f"metrics_by_round.{stage}.png"), dataset)
    plot_curve(curves, task, manifest, data_dir, os.path.join(results_dir, "fitted_curve.png"), dataset)
    print(f"plots written to {results_dir}")


if __name__ == "__main__":
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    repo = os.path.dirname(here)
    fed = os.path.join(repo, "data", "simulated_data", "federated")
    datasets = sys.argv[1:] or sorted(d for d in os.listdir(os.path.join(here, "results"))
                                       if os.path.isfile(os.path.join(here, "results", d, "metrics.csv")))
    for ds in datasets:
        plot_dataset(ds, os.path.join(here, "results", ds), os.path.join(fed, ds))
    print(plot_overview(datasets, os.path.join(here, "results"), fed,
                        os.path.join(here, "results", "fitted_curves_all.png")).to_string(index=False))
