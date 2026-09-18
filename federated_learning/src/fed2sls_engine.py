"""The `fed2sls` method of job.py: exact federated 2SLS from summed sufficient statistics.

Nothing is trained. Each site releases the cross-products of its centred
design (flamingo_fedmr.site_stats), the server sums them and solves; a
second round returns the HC0 robust covariance. Both engines run the same
arithmetic:

    nvflare   src/fed2sls_controller.py on the server, src/fed2sls_client.py at every
              site, in the simulator; the estimate is then checked against the
              in-process protocol on the same files (must agree to TOL)
    local     flamingo_fedmr's protocol classes in this process

For datasets whose sites hold their own SNPs (the repo's default) the
estimate also equals the repo's pooled 2SLS with per-site first stages and
site intercepts, and that difference is checked and recorded too.

Outputs, under results/fed2sls/<dataset>/:
    estimates.<basis>[.cf<k>].json   the server's full result: theta, SE, robust SE, diagnostics
    curves.csv                       the fitted causal curve on the shared X grid with its
                                     analytic 95% band, in the layout plots.py reads
    metrics.csv                      one row: estimates, SEs, first-stage F, identity-check gaps
"""

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

import flamingo_fedmr as fm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tasks import X_GRID  # noqa: E402

TOL = 1e-10
METHOD = "fed2sls"
CURVE_COLUMNS = ["site", "round", "x", "f", "f_lo", "f_hi"]


def spec_tag(basis, crossfit):
    return basis + (f".cf{crossfit}" if crossfit else "")


def protocol_name(manifest):
    return "shared" if manifest.get("shared_snps") else "local"


def supported(manifest, basis, crossfit):
    """Why this (dataset, basis, crossfit) cannot run, or None if it can. The NVFlare transport
    (and so this method) runs the shared-instrument protocol with the linear basis only."""
    if protocol_name(manifest) == "shared" and (basis != "linear" or crossfit):
        return "the shared-instrument protocol runs with basis=linear and no cross-fit"
    return None


def pooled_reference(sites, basis):
    """The repo's concatenated fit for site-specific SNPs, in numpy: per-site first stage
    X ~ [1, G], then one 2SLS on the stacked rows with site dummies, on [X] or [X, X^2]
    instrumented by [xhat] or [xhat, xhat^2]. The arithmetic of pooled_2sls / quadratic_2sls
    in data/scripts/federated_summary_mr.py."""
    K = len(sites)
    Zs, Ws, Ys = [], [], []
    for k, s in enumerate(sites):
        Z1 = np.column_stack([np.ones(s.n), s.G])
        xhat = Z1 @ np.linalg.lstsq(Z1, s.X, rcond=None)[0]
        S = np.zeros((s.n, K))
        S[:, k] = 1
        if basis == "quadratic":
            Zs.append(np.column_stack([xhat, xhat**2, S]))
            Ws.append(np.column_stack([s.X, s.X**2, S]))
        else:
            Zs.append(np.column_stack([xhat, S]))
            Ws.append(np.column_stack([s.X, S]))
        Ys.append(s.Y)
    Z, W, Y = np.vstack(Zs), np.vstack(Ws), np.concatenate(Ys)
    P = Z @ np.linalg.lstsq(Z, W, rcond=None)[0]
    theta = np.linalg.lstsq(P, Y, rcond=None)[0]
    return {"X": float(theta[0]), "X2": float(theta[1])} if basis == "quadratic" else {"X": float(theta[0])}


def run_in_process(data, protocol, basis, crossfit, seed=0):
    """The protocol on loaded sites; returns (estimates dict as the controller writes it, FedMRResult)."""
    cls = fm.SharedInstrumentFedMR if protocol == "shared" else fm.LocalFirstStageFedMR
    run = cls(basis=basis, crossfit=crossfit, robust=True, seed=seed).run(data)
    spec = {"protocol": protocol, "basis": basis, "crossfit": crossfit, "seed": seed}
    out = {"sites": run.stats.layout.sites, "rounds": run.rounds, "spec": spec, **run.result.to_dict()}
    return out, run.result


def run_nvflare(data_dir, sites, protocol, basis, crossfit, seed, out_path, workspace, simulator_run,
                task_interval):
    """Fed2SLSController on the server, fed2sls_client.py at every site, through the simulator."""
    from nvflare.job_config.api import FedJob
    from nvflare.job_config.script_runner import FrameworkType, ScriptRunner
    from fed2sls_controller import Fed2SLSController

    here = os.path.dirname(os.path.abspath(__file__))
    job = FedJob(name=f"fed2sls_{os.path.basename(data_dir)}_{basis}", min_clients=len(sites))
    job.to_server(Fed2SLSController(protocol=protocol, basis=basis, crossfit=crossfit, robust=True, seed=seed,
                                  out_path=out_path))
    # framework NUMPY: the default PyTorch converters expect tensors and drop plain numpy payloads
    runner = ScriptRunner(script=os.path.join(here, "fed2sls_client.py"),
                          script_args=f"--data_dir {data_dir} --sites {','.join(sites)}", framework=FrameworkType.NUMPY)
    for site in sites:
        job.to(runner, site)
    simulator_run(job, workspace, sites, len(sites), task_interval)
    with open(out_path) as f:
        return json.load(f)


def curve_table(est, rounds):
    """theta' b(x) on the X grid with its analytic 95% band, b(x) = [x] or [x, x^2]. No intercept:
    every estimate is within site, so the curve is relative to X = 0, as plots.py draws it."""
    names = est["w_names"]
    endog = [n for n in names if n in ("X", "X2")]
    idx = [names.index(n) for n in endog]
    theta = np.array(est["theta"])[idx]
    se = np.array(est["robust_se"] if est.get("robust_se") else est["se"])[idx]
    x = np.asarray(X_GRID, dtype=float)
    B = np.column_stack([x if n == "X" else x**2 for n in endog])
    f = B @ theta
    # covariance across the endogenous coefficients is not in the JSON; the band uses the
    # diagonal, which for the linear basis is exact
    half = 1.96 * np.sqrt((B**2) @ se**2)
    return pd.DataFrame({"site": "all", "round": rounds, "x": x, "f": f, "f_lo": f - half, "f_hi": f + half})[CURVE_COLUMNS]


def metrics_row(est, protocol, basis, crossfit, diffs):
    names = est["w_names"]
    fs = est["diagnostics"]["first_stage"].get("X", {})
    row = {"site": "all", "round": est["rounds"], "stage": "global", "n_train": est["N"], "n_test": 0,
           "protocol": protocol, "basis": basis, "crossfit": crossfit, "N": est["N"]}
    for n in ("X", "X2"):
        if n in names:
            i = names.index(n)
            row[f"theta_{n}"] = est["theta"][i]
            row[f"se_{n}"] = est["se"][i]
            row[f"robust_se_{n}"] = est["robust_se"][i] if est.get("robust_se") else float("nan")
    row["first_stage_F"] = fs.get("F", float("nan"))
    row["partial_r2"] = fs.get("partial_r2", float("nan"))
    row.update(diffs)
    return row


def run(dataset, data_dir, results_dir, workspace, engine, basis, crossfit, seed, simulator_run, task_interval):
    """Run the method on one dataset with the chosen engine, write the three outputs, return the
    metrics row. Raises SystemExit when an identity check fails."""
    with open(os.path.join(data_dir, "manifest.json")) as f:
        manifest = json.load(f)
    protocol = protocol_name(manifest)
    sites = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(data_dir, "site*.csv")))
    data = fm.load_sites(data_dir)
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, f"estimates.{spec_tag(basis, crossfit)}.json")

    reference, ref_result = run_in_process(data, protocol, basis, crossfit, seed)
    diffs = {}
    if engine == "nvflare":
        est = run_nvflare(data_dir, sites, protocol, basis, crossfit, seed, out_path, workspace, simulator_run,
                          task_interval)
        theta = dict(zip(est["w_names"], est["theta"]))
        rse = dict(zip(est["w_names"], est["robust_se"]))
        diffs["max_diff_vs_inprocess"] = max(abs(theta[n] - ref_result[n]) for n in ref_result.w_names)
        diffs["max_diff_robust_se_vs_inprocess"] = max(abs(rse[n] - ref_result.se(n, True)) for n in ref_result.w_names)
    else:
        est = reference
        with open(out_path, "w") as f:
            json.dump(est, f, indent=1)
    theta = dict(zip(est["w_names"], est["theta"]))
    if protocol == "local" and not crossfit:
        pooled = pooled_reference(data, basis)
        diffs["max_diff_vs_pooled"] = max(abs(theta[n] - pooled[n]) for n in pooled)

    curve_table(est, est["rounds"]).to_csv(os.path.join(results_dir, "curves.csv"), index=False)
    row = metrics_row(est, protocol, basis, crossfit, diffs)
    pd.DataFrame([row]).to_csv(os.path.join(results_dir, "metrics.csv"), index=False)

    endog = [n for n in est["w_names"] if n in ("X", "X2")]
    print(f"Fed-2SLS fit ({engine}, protocol={protocol}, basis={basis}"
          + (f", crossfit={crossfit}" if crossfit else "") + f") over {len(sites)} sites, N = {est['N']}, "
          f"{est['rounds']} rounds:")
    for n in endog:
        i = est["w_names"].index(n)
        print(f"  {n:2s} = {est['theta'][i]:.6f}  se {est['se'][i]:.6f}  robust se {est['robust_se'][i]:.6f}")
    print(f"  first-stage F = {row['first_stage_F']:.1f}  partial R2 = {row['partial_r2']:.4f}")
    if diffs:
        print("  identity checks: " + "  ".join(f"{k} = {v:.2e}" for k, v in diffs.items()))
    if any(v >= TOL for v in diffs.values()):
        raise SystemExit(f"Fed-2SLS identity check failed for {dataset}: {diffs}")
    print(f"wrote {results_dir}", flush=True)
    return row
