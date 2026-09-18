"""Plots for the federated runs, from results/<method>/<dataset>/{metrics,curves}.csv.

    uv run python src/plots.py                    # every method and dataset found + overview
    uv run python src/plots.py 2sri quadratic     # one method / dataset

results/<method>/<dataset>/metrics_by_round.{global,local}.png
    one panel per metric across rounds; gray lines are the
    sites, bold blue the test-size-weighted mean.
results/<method>/<dataset>/fitted_curve.png
    the causal-curve head f(X) at the last round against the true curve from
    manifest.json, the pooled binned mean of the outcome, and the naive
    federated fit when results/naive/<dataset> exists.
results/fitted_curves_all.png, results/summary.csv
    overview across datasets and methods.
"""

import os
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                "data", "scripts"))
from tasks import detect_task, load_manifest, true_curve  # noqa: E402
import mr_style as S  # noqa: E402  one look per estimator across every figure in the repo

BLUE, GRAY, INK, MUTED, SURFACE, GRID = S.BLUE, S.DATA_GREY, S.INK, S.MUTED, S.SURFACE, S.GRID
METHOD_STYLE = {                     # colour and line style per method, shared with the MR scripts
    "naive": dict(**S.NAIVE, label="federated naive: f(X)"),
    "2sri": dict(**S.FED2SRI, label="federated Fed-2SRI: f(X) with control function"),
    "2sps": dict(**S.FED2SPS, label="federated Fed-2SPS: f(X_hat)"),
    "fed2sls": dict(**S.FED2SLS, label="federated Fed-2SLS: exact pooled 2SLS, 95% band"),
}
METHODS = list(METHOD_STYLE)


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, lw=0.8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#cfcfcc")
    ax.tick_params(colors=MUTED, labelsize=8, length=0)


def last_curve(curves):
    last = curves["round"].max()
    c = curves[curves["round"] == last].groupby("x")["f"].mean().sort_index()
    x, f = c.index.to_numpy(), c.to_numpy()
    f = f - np.interp(0.0, x, f)                 # anchor at X = 0, like the true curve
    return x, f, int(last)


def last_band(curves):
    """(x, f_lo, f_hi) of the last round, anchored like last_curve, or None when the run
    carries no analytic band (the FedAvg methods)."""
    if not {"f_lo", "f_hi"} <= set(curves.columns):
        return None
    last = curves["round"].max()
    c = curves[curves["round"] == last].groupby("x")[["f", "f_lo", "f_hi"]].mean().sort_index()
    x = c.index.to_numpy()
    f0 = np.interp(0.0, x, c.f.to_numpy())
    return x, c.f_lo.to_numpy() - f0, c.f_hi.to_numpy() - f0


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def plot_metrics(df, task, stage, out, dataset, method):
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
    fig.suptitle(f"{dataset} / {method}: per-site test metrics per round ({label})", x=0.01, ha="left",
                 fontsize=12, color=INK, fontweight="bold")
    fig.text(0.01, 0.93 if nrow > 1 else 0.88,
             "Gray: each site on its own 20% test split.  Blue: test-size-weighted mean across sites.",
             fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.92 if nrow > 1 else 0.86))
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def draw_curves(ax, method_runs, task, manifest, data_dir, title):
    """method_runs: {method: (metrics, curves)}. Draws truth, binned means, one line per method.
    Binary outcomes are drawn on the logit scale, where the model's f lives."""
    _style(ax)
    ax.grid(axis="x", color=GRID, lw=0.8)
    if task.name != "survival":
        pooled = pd.concat(pd.read_csv(os.path.join(data_dir, f)) for f in sorted(os.listdir(data_dir))
                           if f.startswith("site") and f.endswith(".csv"))
        pooled["bin"] = pd.cut(pooled.X, np.linspace(-3, 3, 25))
        emp = pooled.groupby("bin", observed=True).agg(x=("X", "mean"), y=("Y", "mean"), n=("Y", "size"))
        emp = emp[emp.n >= 30]
        y = emp.y.to_numpy() if task.name == "continuous" else _logit(emp.y.to_numpy())
        ylab = "mean Y" if task.name == "continuous" else "logit of P(Y = 1)"
        y0 = np.interp(0.0, emp.x, y)
        ax.scatter(emp.x, y - y0, s=18, color=GRAY, zorder=2,
                   label=f"pooled data: binned {ylab}, relative to X = 0")
    x_ref = None
    for method in METHODS:
        if method not in method_runs:
            continue
        metrics, curves = method_runs[method]
        x, f, last = last_curve(curves)
        x_ref = x
        st = METHOD_STYLE[method]
        if method == "2sps" and "xhat_sd" in metrics.columns:
            # f(X_hat) is only identified where X_hat has support: full line within 2 sd, faint beyond
            lim = 2 * metrics.xhat_sd.max()
            inside = np.abs(x) <= lim
            ax.plot(x[inside], f[inside], **S.line(st), zorder=4,
                    label=f'{st["label"]}, round {last} (|X| <= 2 sd of X_hat = {lim:.1f})')
            ax.plot(x, f, color=st["color"], lw=1.0, ls=":", zorder=3)
            continue
        if method == "fed2sls":
            band = last_band(curves)
            if band is not None:
                ax.fill_between(band[0], band[1], band[2], color=st["color"], alpha=st["band_alpha"], lw=0, zorder=2)
            ax.plot(x, f, **S.line(st), zorder=4, label=st["label"])
            continue
        ax.plot(x, f, **S.line(st), zorder=4, label=f'{st["label"]}, round {last}')
    tc = true_curve(manifest, x_ref)
    if tc is not None:
        ax.plot(x_ref, tc - np.interp(0.0, x_ref, tc), **S.line(S.TRUTH), zorder=3, label="true causal curve f(X)")
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.set_xlabel("X (exposure)", fontsize=9, color=MUTED)
    ax.set_ylabel(task.curve_label + ", relative to X = 0", fontsize=9, color=MUTED)


NOTE = ("naive fits E[outcome | X], which carries the confounder U's path; the MR methods use the SNPs "
        "as instruments so f(X) targets the causal curve.")
METHOD_TITLE = {"naive": "federated naive", "2sri": "Fed-2SRI", "2sps": "Fed-2SPS", "fed2sls": "Fed-2SLS"}
METHOD_NOTE = {           # one line under the title of the per-method plot
    "naive": "outcome ~ f(X) with no instruments: E[outcome | X], which carries the confounder U's path.",
    "2sri": "site-local first stage X ~ SNPs, then a federated network f(X) + c (X - X_hat) trained with FedAvg; "
            "f is the causal curve and the control-function term absorbs the confounder.",
    "2sps": "site-local first stage X ~ SNPs, then a federated network f(X_hat) trained with FedAvg; "
            "solid where X_hat has support.",
    "fed2sls": "exact pooled 2SLS from summed sufficient statistics: no training, two rounds; "
               "the band is the analytic 95% interval.",
}


def _load(results_root, method, dataset):
    d = os.path.join(results_root, method, dataset)
    if not os.path.isfile(os.path.join(d, "curves.csv")):
        return None
    return pd.read_csv(os.path.join(d, "metrics.csv")), pd.read_csv(os.path.join(d, "curves.csv"))


def plot_dataset(dataset, method, results_root, data_dir, task=None):
    manifest = load_manifest(data_dir)
    if task is None:
        task = detect_task(pd.read_csv(os.path.join(data_dir, "site01.csv"), nrows=2000), manifest)
    results_dir = os.path.join(results_root, method, dataset)
    metrics, curves = _load(results_root, method, dataset)
    if method != "fed2sls":       # fed2sls has no per-round test metrics: it is one closed-form solve
        for stage in ("global", "local"):
            plot_metrics(metrics, task, stage, os.path.join(results_dir, f"metrics_by_round.{stage}.png"),
                         dataset, method)

    fig, ax = plt.subplots(figsize=(7.5, 4.8), facecolor=SURFACE)
    draw_curves(ax, {method: (metrics, curves)}, task, manifest, data_dir, "")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle(f"{dataset}: fitted X -> outcome curve, {METHOD_TITLE.get(method, method)}", x=0.01, ha="left",
                 fontsize=12, color=INK, fontweight="bold")
    fig.text(0.01, 0.9, METHOD_NOTE.get(method, ""), fontsize=8, color=MUTED, wrap=True)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(os.path.join(results_dir, "fitted_curve.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"plots written to {results_dir}")


def plot_overview(datasets, results_root, fed_dir, out):
    """One panel per dataset with every method's causal-curve head, plus summary.csv of
    last-round weighted test metrics per method and dataset."""
    ncol = max(1, min(3, len(datasets)))
    nrow = int(np.ceil(len(datasets) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol + 1.5, 3.8 * nrow + 1.6), facecolor=SURFACE,
                             squeeze=False)
    for ax in axes.flat[len(datasets):]:
        ax.axis("off")
    rows = []
    for ax, ds in zip(axes.flat, datasets):
        data_dir = os.path.join(fed_dir, ds)
        manifest = load_manifest(data_dir)
        task = detect_task(pd.read_csv(os.path.join(data_dir, "site01.csv"), nrows=2000), manifest)
        method_runs = {}
        for method in METHODS:
            loaded = _load(results_root, method, ds)
            if loaded is None:
                continue
            metrics, curves = loaded
            method_runs[method] = loaded
            if method == "fed2sls":
                r = metrics.iloc[-1]
                rows.append({"dataset": ds, "method": method, "task": task.name, "round": int(r["round"]),
                             **{k: r[k] for k in ("theta_X", "se_X", "robust_se_X", "first_stage_F") if k in r}})
                continue
            g = metrics[(metrics["round"] == metrics["round"].max()) & (metrics.stage == "global")]
            w = g.n_test
            score = {m: (g[m] * w).sum() / w.sum() for m in task.metrics if m not in ("loss", "events")}
            if "fs_r2" in g.columns:
                score["fs_r2"] = (g.fs_r2 * w).sum() / w.sum()
            rows.append({"dataset": ds, "method": method, "task": task.name, "round": int(g["round"].iloc[0]), **score})
        draw_curves(ax, method_runs, task, manifest, data_dir, f"{ds} ({task.name})")
    handles = {}
    for ax in axes.flat:
        for h, l in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(l.split(", round")[0].split(" (solid")[0], h)
    legend_cols = 1 if ncol == 1 else 2
    legend_rows = int(np.ceil(len(handles) / legend_cols))
    fig.legend(handles.values(), handles.keys(), frameon=False, fontsize=8, loc="lower center", ncol=legend_cols,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Federated X -> outcome curves" if ncol == 1 else
                 "Federated X -> outcome curve: naive vs MR (Fed-2SRI, Fed-2SPS, Fed-2SLS)",
                 x=0.01, ha="left", fontsize=12, color=INK, fontweight="bold")
    fig.text(0.01, 0.945 if nrow > 1 else 0.92, textwrap.fill(NOTE, width=int(17 * (4.6 * ncol + 1.5))),
             fontsize=8, color=MUTED, va="top")
    bottom = (0.25 + 0.16 * legend_rows) / (3.8 * nrow + 1.6)     # inches of legend as a figure fraction
    fig.tight_layout(rect=(0, bottom, 1, 0.93 if nrow > 1 else 0.9))
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(results_root, "summary.csv"), index=False)
    return summary


if __name__ == "__main__":
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fed = os.path.join(here, "..", "data", "simulated_data", "federated")
    root = os.path.join(here, "results")
    if len(sys.argv) == 3:
        pairs = [(sys.argv[1], sys.argv[2])]
    else:
        pairs = [(m, d) for m in METHODS if os.path.isdir(os.path.join(root, m))
                 for d in sorted(os.listdir(os.path.join(root, m)))
                 if os.path.isfile(os.path.join(root, m, d, "metrics.csv"))]
    for m, d in pairs:
        plot_dataset(d, m, root, os.path.join(fed, d))
    datasets = sorted({d for _, d in pairs})
    print(plot_overview(datasets, root, fed, os.path.join(root, "fitted_curves_all.png")).to_string(index=False))
