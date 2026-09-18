"""Sweep robust / private federation scenarios with the local engine and summarise them.

    uv run python secure_experiments.py --suite robustness --jobs 12
    uv run python secure_experiments.py --suite all --jobs 12
    uv run python secure_experiments.py --suite privacy --report_only     # re-plot from runs.csv

Suites and scenarios are defined in configs/secure/experiments.yaml. Every
scenario runs per dataset and seed; every suite also runs "clean" (the
original FedAvg code path, fedsec off) as the reference. Per run it records:

    rmse_truth     RMSE of the final global causal-curve head f against the true curve, both
                   anchored at X = 0, on |X| <= 2 (NaN for binary: no truth on the logit scale)
    rmse_clean     the same against the clean run with the same dataset and seed: the part of
                   the error the attack / defense / privacy mechanism adds
    score          honest sites' test-size-weighted test metric of the final aggregate
                   (r2, auc, or c_index)
    tpr, fpr       share of malicious / honest site-rounds the server flagged (attack rounds)
    update_cos     honest sites: cosine(message the server sees, true update)
    mia_auc        honest sites: membership-inference AUC from the message
    eps_aggregate, eps_server   from the accountant (inf = no dp protection)

Writes results/secure/experiments/<suite>/{runs.csv, summary.csv, final_curves.csv, *.png};
raw per-run logs go to workspace/secure_experiments/ (git-ignored).
"""

import argparse
import contextlib
import copy
import glob
import json
import math
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
FED_DIR = os.path.join(os.path.dirname(HERE), "data", "simulated_data", "federated")
OUT_ROOT = os.path.join(HERE, "results", "secure", "experiments")
sys.path.insert(0, os.path.join(HERE, "src"))

SCORE = {"continuous": "r2", "binary": "auc", "survival": "c_index"}
X_MAX = 2.0


def deep_merge(a, b):
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def suite_scenarios(suite):
    """{scenario name: fedsec config dict}; 'clean' first."""
    scen = {"clean": None}
    if "attacks" in suite:
        for an, a in suite["attacks"].items():
            for dn, d in suite["defenses"].items():
                scen[f"{an}|{dn}"] = deep_merge(a, d)
    scen.update(suite.get("scenarios", {}))
    return scen


def final_curve(curves):
    last = curves["round"].max()
    c = curves[curves["round"] == last].groupby("x")["f"].mean().sort_index()
    x, f = c.index.to_numpy(), c.to_numpy()
    return x, f - np.interp(0.0, x, f)


def run_one(job):
    """One (scenario, dataset, seed) run in a worker process. Returns (row, curve rows)."""
    import local_engine
    from fedsec.config import from_dict
    from tasks import detect_task, load_manifest, true_curve

    data_dir = os.path.join(FED_DIR, job["dataset"])
    sites = sorted(os.path.basename(f)[:-4] for f in glob.glob(os.path.join(data_dir, "site*.csv")))
    out = job["out_dir"]
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    cfg = from_dict({**(job["config"] or {}), "name": job["scenario"]}) if job["config"] is not None else None
    t0 = time.time()
    with open(os.path.join(out, "run.log"), "w") as log, contextlib.redirect_stdout(log):
        local_engine.run(sites, data_dir, out, job["method"], job["rounds"], job["epochs"], job["lr"],
                         seed=job["seed"], secure=cfg)
    secs = time.time() - t0

    manifest = load_manifest(data_dir)
    metrics = pd.concat(pd.read_csv(f) for f in sorted(glob.glob(os.path.join(out, "*.csv"))))
    curves = pd.concat(pd.read_csv(f) for f in sorted(glob.glob(os.path.join(out, "curves", "*.csv"))))
    task = detect_task(pd.read_csv(os.path.join(data_dir, f"{sites[0]}.csv"), nrows=2000), manifest)
    fs = os.path.join(out, "fedsec")
    priv = json.load(open(os.path.join(fs, "privacy.json"))) if os.path.isfile(os.path.join(fs, "privacy.json")) else {}
    bad = set(priv.get("malicious_sites", []))

    x, f = final_curve(curves)
    keep = np.abs(x) <= X_MAX
    tc = true_curve(manifest, x)
    rmse_truth = float("nan") if tc is None else \
        float(np.sqrt(np.mean((f[keep] - (tc - np.interp(0.0, x, tc))[keep]) ** 2)))
    last = metrics[(metrics["round"] == metrics["round"].max()) & (metrics.stage == "global")]
    honest = last[~last.site.isin(bad)]
    metric = SCORE[task.name]
    score = float((honest[metric] * honest.n_test).sum() / honest.n_test.sum())

    row = {"suite": job["suite"], "scenario": job["scenario"], "dataset": job["dataset"], "seed": job["seed"],
           "task": task.name, "method": job["method"], "rmse_truth": rmse_truth, "score_metric": metric,
           "score": score, "tpr": float("nan"), "fpr": float("nan"), "update_cos": float("nan"),
           "mia_auc": float("nan"), "eps_aggregate": math.inf, "eps_server": math.inf,
           "n_malicious": len(bad), "seconds": secs}
    if priv:
        row["eps_aggregate"] = float(priv["epsilon_aggregate"])
        row["eps_server"] = float(priv["epsilon_server"])
    sp, cp = os.path.join(fs, "server.csv"), os.path.join(fs, "clients.csv")
    if os.path.isfile(sp) and bad:
        s = pd.read_csv(sp)
        start = (job["config"] or {}).get("attack", {}).get("start_round", 0)
        s = s[s["round"] >= start]
        row["tpr"] = float(s[s.malicious == 1].flagged.mean())
        row["fpr"] = float(s[s.malicious == 0].flagged.mean())
    if os.path.isfile(cp):
        c = pd.read_csv(cp)
        h = c[c.malicious == 0]
        row["update_cos"] = float(h.update_cos.mean())
        row["mia_auc"] = float(h.mia_auc.mean())
    curve_rows = pd.DataFrame({"suite": job["suite"], "scenario": job["scenario"], "dataset": job["dataset"],
                               "seed": job["seed"], "x": x, "f": f})
    return row, curve_rows


def add_rmse_clean(runs, curves):
    clean = curves[curves.scenario == "clean"][["dataset", "seed", "x", "f"]].rename(columns={"f": "f_clean"})
    m = curves.merge(clean, on=["dataset", "seed", "x"])
    m = m[np.abs(m.x) <= X_MAX]
    r = (m.assign(d2=(m.f - m.f_clean) ** 2).groupby(["scenario", "dataset", "seed"]).d2.mean() ** 0.5)
    return runs.drop(columns=["rmse_clean"], errors="ignore").merge(
        r.rename("rmse_clean").reset_index(), on=["scenario", "dataset", "seed"], how="left")


def summarize(runs):
    cols = ["rmse_truth", "rmse_clean", "score", "tpr", "fpr", "update_cos", "mia_auc"]
    g = runs.groupby(["suite", "scenario", "dataset"], sort=False)
    out = g[cols].agg(["mean", "std"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    eps = g[["eps_aggregate", "eps_server", "n_malicious"]].first()
    return pd.concat([out, eps, g.size().rename("n_seeds")], axis=1).reset_index()


def run_suite(name, spec, defaults, args):
    d = {**defaults, **{k: v for k, v in spec.items() if k in defaults}}
    datasets = args.dataset or d["datasets"]
    seeds = args.seeds if args.seeds is not None else d["seeds"]
    scen = suite_scenarios(spec)
    if args.scenario:
        scen = {k: v for k, v in scen.items() if k == "clean" or any(s in k for s in args.scenario)}
    out_dir = os.path.join(OUT_ROOT, name)
    raw = os.path.join(args.workspace, name)
    jobs = [{"suite": name, "scenario": sn, "config": sc, "dataset": ds, "seed": sd, "method": d["method"],
             "rounds": d["rounds"], "epochs": d["epochs"], "lr": d["lr"],
             "out_dir": os.path.join(raw, sn.replace("|", "__"), ds, f"seed{sd}")}
            for sn, sc in scen.items() for ds in datasets for sd in seeds]
    # fail on an invalid scenario before running anything
    from fedsec.config import from_dict, validate
    sites10 = [f"site{i:02d}" for i in range(1, 11)]
    for sn, sc in scen.items():
        if sc is not None:
            validate(from_dict(sc), sites10, warn=lambda *_: None)
    print(f"[{name}] {len(scen)} scenarios x {len(datasets)} datasets x {len(seeds)} seeds = {len(jobs)} runs, "
          f"{args.jobs} at a time", flush=True)
    rows, curves, t0 = [], [], time.time()
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(run_one, j): j for j in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            j = futs[fut]
            try:
                row, cr = fut.result()
            except Exception as e:  # keep the sweep going; report at the end
                print(f"  FAILED {j['scenario']} {j['dataset']} seed{j['seed']}: {e!r}", flush=True)
                continue
            rows.append(row)
            curves.append(cr)
            if i % 20 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} done ({time.time() - t0:.0f} s)", flush=True)
    if len(rows) != len(jobs):
        raise SystemExit(f"[{name}] {len(jobs) - len(rows)} run(s) failed; logs under {raw}")
    runs = pd.DataFrame(rows)
    curves = pd.concat(curves, ignore_index=True)
    runs = add_rmse_clean(runs, curves)
    os.makedirs(out_dir, exist_ok=True)
    order = {s: i for i, s in enumerate(scen)}
    runs = runs.sort_values(["dataset", "scenario", "seed"], key=lambda c: c.map(order) if c.name == "scenario" else c)
    runs.to_csv(os.path.join(out_dir, "runs.csv"), index=False)
    curves.to_csv(os.path.join(out_dir, "final_curves.csv"), index=False)
    summary = summarize(runs)
    summary.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    with open(os.path.join(out_dir, "experiment.json"), "w") as f:
        json.dump({"suite": name, "settings": d, "datasets": datasets, "seeds": seeds,
                   "scenarios": {k: v for k, v in scen.items()}}, f, indent=1)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--suite", default="all", help="suite name in the experiments file, or 'all'")
    p.add_argument("--experiments", default=os.path.join(HERE, "configs", "secure", "experiments.yaml"))
    p.add_argument("--dataset", action="append", help="override the suite's datasets (repeatable)")
    p.add_argument("--seeds", type=int, nargs="+", help="override the suite's seeds")
    p.add_argument("--scenario", action="append", help="only scenarios whose name contains this (repeatable)")
    p.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    p.add_argument("--workspace", default=os.path.join(HERE, "workspace", "secure_experiments"))
    p.add_argument("--report_only", action="store_true", help="skip the runs; re-plot from runs.csv")
    args = p.parse_args()

    spec = yaml.safe_load(open(args.experiments))
    names = list(spec["suites"]) if args.suite == "all" else [args.suite]
    from fedsec import report
    for name in names:
        if name not in spec["suites"]:
            raise SystemExit(f"no suite {name!r}; have {list(spec['suites'])}")
        if not args.report_only:
            run_suite(name, spec["suites"][name], spec["defaults"], args)
        out_dir = os.path.join(OUT_ROOT, name)
        report.make(name, out_dir, FED_DIR)
        print(f"[{name}] wrote {os.path.relpath(out_dir, HERE)}/", flush=True)


if __name__ == "__main__":
    main()
