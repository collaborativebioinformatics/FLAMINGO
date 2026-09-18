"""Figures for secure_experiments.py, in the style of src/plots.py.

robustness   robustness_heatmap.png   RMSE of the causal curve vs truth and vs clean, attack x defense
             robustness_curves.<dataset>.png   final causal curves under each attack, four aggregation rules
privacy      privacy_utility.png      RMSE vs epsilon for central / distributed+secagg / local dp
             leakage.png              what the server sees of one site's update, per protection
combined     combined.png             RMSE under a sign-flip attacker for privacy designs
"""

import os

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

import plots  # noqa: E402
from tasks import load_manifest, true_curve  # noqa: E402

INK, MUTED, SURFACE, GRID, GRAY = plots.INK, plots.MUTED, plots.SURFACE, plots.GRID, plots.GRAY
# categorical slots in fixed order (the palette src/plots.py draws from)
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
X_MAX = 2.0


def _save(fig, path):
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def _truth(fed_dir, dataset, x):
    tc = true_curve(load_manifest(os.path.join(fed_dir, dataset)), x)
    return None if tc is None else tc - np.interp(0.0, x, tc)


def _fmt(v):
    return "inf" if not np.isfinite(v) else (f"{v:.2g}" if v < 10 else f"{v:.0f}")


def robustness(out_dir, summary, curves, fed_dir):
    s = summary[summary.scenario != "clean"].copy()
    s[["attack", "defense"]] = s.scenario.str.split("|", expand=True)
    attacks = list(dict.fromkeys(s.attack))
    defenses = list(dict.fromkeys(s.defense))
    datasets = list(dict.fromkeys(s.dataset))
    clean = summary[summary.scenario == "clean"].set_index("dataset").rmse_truth_mean

    rows = [("rmse_truth_mean", "vs the true curve", lambda ds: f"clean FedAvg {clean.get(ds, np.nan):.3f}"),
            ("rmse_clean_mean", "vs clean FedAvg, same seed", lambda ds: "added by attack + defense")]
    fig, axes = plt.subplots(2, len(datasets), figsize=(6.2 * len(datasets), 8.6), facecolor=SURFACE, squeeze=False)
    for ax, (ds, (col, what, note)) in zip(axes.flat, [(d, r) for r in rows for d in datasets]):
        m = s[s.dataset == ds].pivot(index="attack", columns="defense", values=col)
        m = m.reindex(index=attacks, columns=defenses)
        vals = m.to_numpy(dtype=float)
        lo, hi = max(np.nanmin(vals), 1e-3), np.nanmax(vals)
        im = ax.imshow(vals, cmap="Blues", norm=LogNorm(vmin=max(lo, 1e-3), vmax=max(hi, lo * 1.01)), aspect="auto")
        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                v = vals[i, j]
                dark = v > 0 and hi > lo and np.log(v / lo) > 0.6 * np.log(hi / lo)
                ax.text(j, i, f"{v:.2f}" if v < 10 else f"{v:.0f}", ha="center", va="center", fontsize=7.5,
                        color="white" if dark else INK)
        ax.set_xticks(range(len(defenses)))
        ax.set_xticklabels([d.replace("_", " ") for d in defenses], rotation=35, ha="right", fontsize=8, color=MUTED)
        ax.set_yticks(range(len(attacks)))
        ax.set_yticklabels([a.replace("_", " ") for a in attacks], fontsize=8, color=MUTED)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(f"{ds}: RMSE {what}  ({note(ds)})", loc="left", fontsize=9.5, color=INK)
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cb.ax.tick_params(labelsize=7, colors=MUTED, length=0)
        cb.outline.set_visible(False)
    fig.suptitle("Robustness: error of the federated causal curve f(X) on |X| <= 2, mean over seeds",
                 x=0.01, ha="left", fontsize=11.5, color=INK, fontweight="bold")
    fig.text(0.01, 0.945, "Rows: what the malicious site(s) do. Columns: how the server aggregates. "
             "Darker = larger error (log scale, per panel).", fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, os.path.join(out_dir, "robustness_heatmap.png"))
    for ds in datasets:
        _attack_curves(out_dir, curves, fed_dir, ds, attacks, defenses)


def _attack_curves(out_dir, curves, fed_dir, ds, attacks, defenses):
    """Final curves under each attack for four aggregation rules, one dataset."""
    shown = ["fedavg", "norm_bound", "median", "geometric_median"]
    shown = [d for d in shown if d in defenses]
    atk = [a for a in attacks if a != "no_attack"]
    ncol = 3
    nrow = int(np.ceil(len(atk) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.5 * nrow + 1.0), facecolor=SURFACE, squeeze=False)
    for ax in axes.flat[len(atk):]:
        ax.axis("off")
    c = curves[(curves.dataset == ds) & (np.abs(curves.x) <= X_MAX)]
    mean = c.groupby(["scenario", "x"]).f.mean()
    cl = mean.loc["clean"]
    x = cl.index.to_numpy()
    tr = _truth(fed_dir, ds, x)
    ref = np.concatenate([cl.to_numpy(), tr if tr is not None else []])
    pad = 0.35 * (ref.max() - ref.min())
    for ax, a in zip(axes.flat, atk):
        plots._style(ax)
        ax.plot(x, cl.to_numpy(), color=GRAY, lw=3.2, zorder=1, label="clean FedAvg (no attack)")
        if tr is not None:
            ax.plot(x, tr, color=INK, lw=1.4, ls="--", zorder=5, label="true causal curve")
        for col, d in zip(SLOTS, shown):
            key = f"{a}|{d}"
            if key in mean.index.get_level_values(0):
                ax.plot(x, mean.loc[key].to_numpy(), color=col, lw=2, zorder=3, label=d.replace("_", " "))
        ax.set_ylim(ref.min() - pad, ref.max() + pad)
        ax.set_title(a.replace("_", " "), loc="left", fontsize=10, color=INK)
        ax.set_xlabel("X (exposure)", fontsize=8, color=MUTED)
    h, lab = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, fontsize=8.5, loc="upper right", ncol=3, bbox_to_anchor=(0.99, 0.985))
    fig.suptitle(f"{ds}: final causal curve f(X) under attack (mean over seeds)", x=0.01, ha="left",
                 fontsize=11.5, color=INK, fontweight="bold")
    fig.text(0.01, 0.925 if nrow > 1 else 0.88,
             "Lines that leave the panel are off-scale. Relative to X = 0.", fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.9 if nrow > 1 else 0.84))
    _save(fig, os.path.join(out_dir, f"robustness_curves.{ds}.png"))


MODES = [("central", "central dp (trusted server)"), ("distributed_secagg", "distributed dp + secagg"),
         ("local", "local dp")]


def privacy(out_dir, summary):
    datasets = list(dict.fromkeys(summary.dataset))
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.8 * len(datasets), 4.3), facecolor=SURFACE, squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        plots._style(ax)
        ax.grid(axis="x", color=GRID, lw=0.8)
        d = summary[summary.dataset == ds]
        clean = d[d.scenario == "clean"].rmse_truth_mean.iloc[0]
        ax.axhline(clean, color=GRAY, lw=2, ls="--", zorder=1)
        ax.annotate("no dp (plain FedAvg, secagg): epsilon = inf", (1.0, clean), xycoords=("axes fraction", "data"),
                    xytext=(-4, 5), textcoords="offset points", ha="right", fontsize=7.5, color=MUTED)
        for col, (mode, label) in zip(SLOTS, MODES):
            m = d[d.scenario.str.fullmatch(rf"{mode}_z[\d.]+")].copy()
            if m.empty:
                continue
            m["z"] = m.scenario.str.extract(r"_z([\d.]+)$")[0].astype(float)
            m = m.sort_values("eps_aggregate")
            ax.errorbar(m.eps_aggregate, m.rmse_truth_mean, yerr=m.rmse_truth_std.fillna(0), color=col, lw=2,
                        marker="o", ms=6, capsize=0, elinewidth=1, label=label, zorder=3)
            if mode == MODES[0][0]:
                # the same z gives the same epsilon in every mode: label each z once, along the top
                for _, r in m.iterrows():
                    ax.annotate(f"z={r.z:g}", (r.eps_aggregate, 1.0), xycoords=("data", "axes fraction"),
                                xytext=(0, 2), textcoords="offset points", ha="center", fontsize=7.5, color=MUTED)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.FuncFormatter(
            lambda v, _: f"{v:g}" if f"{v:g}"[0] in "2345" else ""))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.tick_params(axis="x", which="minor", labelsize=7.5, colors=MUTED, length=0)
        ax.set_xlabel("epsilon against observers of the released models (delta = 1e-5, 5 rounds)",
                      fontsize=8, color=MUTED)
        ax.set_ylabel("RMSE of f vs true curve", fontsize=8.5, color=MUTED)
        ax.set_title(ds, loc="left", fontsize=10, color=INK, pad=16)
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, fontsize=8.5, loc="upper right", ncol=3, bbox_to_anchor=(0.99, 0.99))
    fig.suptitle("Privacy vs accuracy of the federated causal curve", x=0.01, ha="left", fontsize=11.5,
                 color=INK, fontweight="bold")
    fig.text(0.01, 0.885, "Site-level dp, clip 0.5; smaller epsilon = stronger guarantee. "
             "Central dp's epsilon does not hold against the server itself; distributed + secagg's does.",
             fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    _save(fig, os.path.join(out_dir, "privacy_utility.png"))

    # what the server sees: one dataset (leakage is a property of the protocol, not the outcome)
    keep = ["plain_fedavg", "central_z1", "distributed_nosecagg_z1", "local_z1", "secagg", "distributed_secagg_z1"]
    # prefer a dataset where the membership-inference probe is defined (not survival)
    ds = next((d for d in datasets if summary[summary.dataset == d].mia_auc_mean.notna().any()), datasets[0])
    d = summary[(summary.dataset == ds) & summary.scenario.isin(keep)].set_index("scenario").reindex(keep).dropna(
        how="all")
    fig, ax = plt.subplots(figsize=(8.2, 3.9), facecolor=SURFACE)
    plots._style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, lw=0.8)
    y = np.arange(len(d))[::-1]
    ax.barh(y, d.update_cos_mean.clip(lower=0), height=0.55, color=SLOTS[0], zorder=2)
    for yi, (name, r) in zip(y, d.iterrows()):
        mia = "" if not np.isfinite(r.mia_auc_mean) else f", MIA AUC {r.mia_auc_mean:.3f}"
        ax.text(max(r.update_cos_mean, 0) + 0.02, yi, f"{r.update_cos_mean:.3f}   (eps vs server "
                f"{_fmt(r.eps_server)}{mia})", va="center", fontsize=8, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([n.replace("_", " ") for n in d.index], fontsize=8.5, color=INK)
    ax.set_xlim(0, 1.75)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("cosine(message the server sees, the site's true update); 1 = the server reads the local model",
                  fontsize=8, color=MUTED)
    fig.suptitle(f"What the server learns about one site's update ({ds}, honest sites, all rounds)", x=0.01,
                 ha="left", fontsize=11, color=INK, fontweight="bold")
    fig.text(0.01, 0.86, "MIA: loss-based membership inference from the message; 0.5 = chance. "
             "z = 1 for the dp rows.", fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    _save(fig, os.path.join(out_dir, "leakage.png"))


def combined(out_dir, summary):
    datasets = list(dict.fromkeys(summary.dataset))
    scen = [s for s in dict.fromkeys(summary.scenario) if s != "clean"]
    fig, axes = plt.subplots(1, len(datasets), figsize=(6.0 * len(datasets), 0.45 * len(scen) + 2.2),
                             facecolor=SURFACE, squeeze=False, sharey=True)
    for ax, ds in zip(axes[0], datasets):
        plots._style(ax)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=GRID, lw=0.8)
        d = summary[summary.dataset == ds].set_index("scenario")
        y = np.arange(len(scen))[::-1]
        vals = d.reindex(scen).rmse_truth_mean.to_numpy()
        ax.barh(y, vals, height=0.55, color=SLOTS[0], zorder=2)
        clean = d.loc["clean", "rmse_truth_mean"]
        ax.axvline(clean, color=GRAY, lw=2, ls="--", zorder=3)
        ax.set_xscale("log")
        for yi, v in zip(y, vals):
            ax.text(v * 1.08, yi, f"{v:.2f}" if v < 10 else f"{v:.0f}", va="center", fontsize=8, color=INK)
        ax.set_yticks(y)
        ax.set_yticklabels([s.replace("_", " ") for s in scen], fontsize=8.5, color=INK)
        ax.set_xlim(right=np.nanmax(vals) * 4)
        ax.set_title(f"{ds} (dashed: clean FedAvg {clean:.2f})", loc="left", fontsize=10, color=INK)
        ax.set_xlabel("RMSE of f vs true curve (log scale)", fontsize=8, color=MUTED)
    fig.suptitle("A sign-flip x10 attacker under each privacy design", x=0.01, ha="left", fontsize=11.5,
                 color=INK, fontweight="bold")
    fig.text(0.01, 0.905, "dp at z = 1, clip 0.5. Local dp alone, with no attack, already gives RMSE ~1.1 "
             "(privacy suite), so the local-dp rows mix both effects.", fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    _save(fig, os.path.join(out_dir, "combined.png"))


def make(suite, out_dir, fed_dir):
    summary = pd.read_csv(os.path.join(out_dir, "summary.csv"))
    curves = pd.read_csv(os.path.join(out_dir, "final_curves.csv"))
    if suite == "robustness":
        robustness(out_dir, summary, curves, fed_dir)
    elif suite == "privacy":
        privacy(out_dir, summary)
    elif suite == "combined":
        combined(out_dir, summary)
