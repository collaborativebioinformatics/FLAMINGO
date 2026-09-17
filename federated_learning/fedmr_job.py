"""Run FedMR as a real federation in the NVFlare simulator, and check it against the
in-process protocol and the pooled fit.

    uv run python fedmr_job.py --dataset linear                 # local first stage, 2 rounds
    uv run python fedmr_job.py --dataset linear_shared          # shared SNPs
    uv run python fedmr_job.py --dataset quadratic --basis quadratic
    uv run python fedmr_job.py --all

One NVFlare client per site CSV. The server (src/fedmr_controller.py) sums the
sites' statistics and solves; the clients (src/fedmr_client.py) compute them.
Nothing is trained. After the job, the NVFlare estimate is compared with
flamingo_fedmr run in-process on the same files; they must agree to 1e-10,
and for datasets with site-specific SNPs so must the repo's pooled 2SLS.

Writes results/fedmr/<dataset>/estimates.json (from the server) and
results/fedmr/summary.csv.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from nvflare.job_config.api import FedJob
from nvflare.job_config.script_runner import FrameworkType, ScriptRunner

import flamingo_fedmr as fm

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
FED_DIR = os.path.join(REPO, "data", "simulated_data", "federated")
sys.path.insert(0, os.path.join(HERE, "src"))
sys.path.insert(0, HERE)
from fedmr_controller import FedMRController  # noqa: E402
from job import simulator_run  # noqa: E402

TOL = 1e-10


def pooled_reference(sites, basis):
    """The repo's concatenated fit for site-specific SNPs, re-done here in numpy (the data scripts
    need polars): per-site first stage X ~ [1, G], then one 2SLS on the stacked rows with site
    dummies, on [X] or [X, X^2] instrumented by [xhat] or [xhat, xhat^2]. This is the arithmetic
    of pooled_2sls / quadratic_2sls in data/scripts/federated_summary_mr.py."""
    K = len(sites)
    Zs, Ws, Ys = [], [], []
    for k, s in enumerate(sites):
        Z1 = np.column_stack([np.ones(s.n), s.G])
        xhat = Z1 @ np.linalg.lstsq(Z1, s.X, rcond=None)[0]
        S = np.zeros((s.n, K)); S[:, k] = 1
        if basis == "quadratic":
            Zs.append(np.column_stack([xhat, xhat**2, S])); Ws.append(np.column_stack([s.X, s.X**2, S]))
        else:
            Zs.append(np.column_stack([xhat, S])); Ws.append(np.column_stack([s.X, S]))
        Ys.append(s.Y)
    Z, W, Y = np.vstack(Zs), np.vstack(Ws), np.concatenate(Ys)
    P = Z @ np.linalg.lstsq(Z, W, rcond=None)[0]
    theta = np.linalg.lstsq(P, Y, rcond=None)[0]
    return {"X": float(theta[0]), "X2": float(theta[1])} if basis == "quadratic" else {"X": float(theta[0])}


def run_dataset(dataset, args):
    data_dir = os.path.join(FED_DIR, dataset)
    manifest = json.load(open(os.path.join(data_dir, "manifest.json")))
    head = pd.read_csv(sorted(glob.glob(os.path.join(data_dir, "site*.csv")))[0], nrows=2000)
    if "Y" not in head.columns or manifest.get("outcome") == "binary" or set(head["Y"].unique()) <= {0, 1}:
        print(f"skip {dataset}: not a continuous outcome (FedMR v1 is linear 2SLS on a continuous Y)", flush=True)
        return None
    sites = sorted(os.path.basename(f)[:-4] for f in glob.glob(os.path.join(data_dir, "site*.csv")))
    protocol = "shared" if manifest.get("shared_snps") else "local"
    out_path = os.path.join(HERE, "results", "fedmr", dataset, f"estimates.{args.basis}.json")
    print(f"\n##### FedMR / {dataset}: protocol={protocol}, basis={args.basis}, {len(sites)} clients #####\n", flush=True)

    job = FedJob(name=f"fedmr_{dataset}_{args.basis}", min_clients=len(sites))
    job.to_server(FedMRController(protocol=protocol, basis=args.basis, crossfit=args.crossfit, robust=True,
                                  out_path=out_path))
    # framework NUMPY: the default PyTorch converters expect tensors and drop plain numpy payloads
    runner = ScriptRunner(script=os.path.join(HERE, "src", "fedmr_client.py"), script_args=f"--data_dir {data_dir}",
                          framework=FrameworkType.NUMPY)
    for site in sites:
        job.to(runner, site)
    workspace = os.path.join(args.workspace, "fedmr", dataset)
    simulator_run(job, workspace, sites, len(sites), args.task_interval)

    nv = json.load(open(out_path))
    nv_theta = dict(zip(nv["w_names"], nv["theta"]))
    nv_rse = dict(zip(nv["w_names"], nv["robust_se"]))

    # the same protocol in-process, from the same files
    data = fm.load_sites(data_dir)
    Proto = fm.SharedInstrumentFedMR if protocol == "shared" else fm.LocalFirstStageFedMR
    local = Proto(basis=args.basis, crossfit=args.crossfit, robust=True).run(data).result
    d_local = max(abs(nv_theta[n] - local[n]) for n in local.w_names)
    d_rse = max(abs(nv_rse[n] - local.se(n, True)) for n in local.w_names)
    line = {"dataset": dataset, "protocol": protocol, "basis": args.basis, "rounds": nv["rounds"], "N": nv["N"],
            **{f"{n}": nv_theta[n] for n in local.w_names}, **{f"se_{n}": nv["se"][i] for i, n in enumerate(local.w_names)},
            **{f"robust_se_{n}": nv_rse[n] for n in local.w_names},
            "max_diff_vs_inprocess": d_local, "max_diff_robust_se_vs_inprocess": d_rse}
    if protocol == "local" and not args.crossfit:
        ref = pooled_reference(data, args.basis)
        line["max_diff_vs_pooled"] = max(abs(nv_theta[n] - ref[n]) for n in ref)
    print(f"NVFlare FedMR: " + "  ".join(f"{n}={nv_theta[n]:.6f} (se {nv['se'][i]:.6f}, robust {nv_rse[n]:.6f})"
                                        for i, n in enumerate(local.w_names)))
    print(f"|NVFlare - in-process| = {d_local:.2e} (robust SE {d_rse:.2e})"
          + (f", |NVFlare - pooled| = {line['max_diff_vs_pooled']:.2e}" if "max_diff_vs_pooled" in line else ""))
    ok = d_local < TOL and d_rse < TOL and line.get("max_diff_vs_pooled", 0.0) < TOL
    print("identity check " + ("PASSED" if ok else "FAILED"), flush=True)
    if not ok:
        raise SystemExit(f"FedMR identity check failed for {dataset}")
    return line


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", action="append", help="subfolder of data/simulated_data/federated/ (repeatable)")
    p.add_argument("--all", action="store_true", help="every continuous dataset")
    p.add_argument("--basis", choices=["linear", "quadratic"], default="linear")
    p.add_argument("--crossfit", type=int, default=0, help="k-fold cross-fitted instrument (local protocol only)")
    p.add_argument("--workspace", default=os.path.join(HERE, "workspace"))
    p.add_argument("--task_interval", type=float, default=0.05)
    args = p.parse_args()
    if args.all:
        datasets = sorted(d for d in os.listdir(FED_DIR) if os.path.isfile(os.path.join(FED_DIR, d, "manifest.json")))
    else:
        datasets = args.dataset or ["linear"]
    lines = [ln for ln in (run_dataset(d, args) for d in datasets) if ln]
    summary_path = os.path.join(HERE, "results", "fedmr", "summary.csv")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    new = pd.DataFrame(lines)
    if os.path.exists(summary_path):
        old = pd.read_csv(summary_path)
        old = old[~(old.dataset.isin(new.dataset) & (old.basis == args.basis))]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(summary_path, index=False)
    pd.set_option("display.width", 200)
    print("\nresults/fedmr/summary.csv:")
    print(new.to_string(index=False))


if __name__ == "__main__":
    main()
