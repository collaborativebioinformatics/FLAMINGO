"""Build and run the NVFlare FedAvg job in the local simulator.

    uv run python job.py --dataset quadratic                 # 2SRI MR on one dataset
    uv run python job.py --all --method naive --method 2sri  # every dataset, two methods
    uv run python job.py --help

Methods (src/model.py):
  naive  fit outcome on X directly: the confounded association
  2sri   site-local OLS first stage X ~ SNPs, then federated outcome ~ f(X) + h(X - X_hat);
         f is the causal curve, h the control function (default)
  2sps   site-local first stage, then federated outcome ~ f(X_hat)
The first stage stays local because every simulated site has its own SNP
effects and allele frequencies, so there is no shared instrument to learn.

One simulated client per site CSV in data/simulated_data/federated/<dataset>/.
The outcome family is detected from the data (src/tasks.py). Each run prints
per-round tables of every client's test metrics and writes
results/<method>/<dataset>/{metrics,curves}.csv plus plots.
"""

import argparse
import glob
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from nvflare.app_opt.pt.job_config.fed_avg import FedAvgJob
from nvflare.job_config.script_runner import ScriptRunner

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
FED_DIR = os.path.join(REPO, "data", "simulated_data", "federated")
sys.path.insert(0, os.path.join(HERE, "src"))
from model import MLP, MRModel  # noqa: E402
from tasks import detect_task, load_manifest  # noqa: E402
import local_engine  # noqa: E402
import plots  # noqa: E402


def simulator_run(job, workspace, clients, threads, task_request_interval=0.05):
    """Like FedAvgJob.simulator_run, but with the server's task_request_interval set.

    NVFlare's server tells every client to wait `task_request_interval` seconds
    (default 2) before asking for its next task, and the simulator sleeps that
    long after each task, so each round costs at least 2 s of idle time. For a
    model that trains in milliseconds that idle time is most of the run.
    """
    with tempfile.TemporaryDirectory() as job_root:
        job.export_job(job_root)
        cfg_path = os.path.join(job_root, job.name, "app_server", "config", "config_fed_server.json")
        cfg = json.load(open(cfg_path))
        cfg.setdefault("server", {})["task_request_interval"] = task_request_interval
        json.dump(cfg, open(cfg_path, "w"), indent=2)
        cmd = (f"{sys.executable} -m nvflare.private.fed.app.simulator.simulator {os.path.join(job_root, job.name)} "
               f"-w {workspace} -c {','.join(clients)} -t {threads}")
        rc = subprocess.run(shlex.split(cmd), env=os.environ.copy()).returncode
        if rc != 0:
            raise SystemExit(f"nvflare simulator exited with {rc}")


def run_dataset(dataset, method, args):
    data_dir = os.path.join(FED_DIR, dataset)
    sites = sorted(os.path.basename(f)[:-4] for f in glob.glob(os.path.join(data_dir, "site*.csv")))
    if not sites:
        raise SystemExit(f"no site*.csv files in {data_dir}")
    head = pd.read_csv(os.path.join(data_dir, f"{sites[0]}.csv"), nrows=2000)
    task = detect_task(head, load_manifest(data_dir))
    print(f"\n##### {dataset} / {method} ({args.engine}): task={task.name}, {len(sites)} sites, "
          f"{args.rounds} rounds x {args.epochs} local epochs #####\n", flush=True)

    workspace = os.path.join(args.workspace, method, dataset)
    metrics_dir = os.path.join(workspace, "metrics")
    shutil.rmtree(workspace, ignore_errors=True)

    if args.engine == "local":
        local_engine.run(sites, data_dir, metrics_dir, method, args.rounds, args.epochs, args.lr, args.batch_size)
    else:
        initial = MLP() if method == "naive" else MRModel(method)
        job = FedAvgJob(name=f"fedavg_{method}_{dataset}", n_clients=len(sites), num_rounds=args.rounds,
                        initial_model=initial, key_metric=task.key_metric)
        runner = ScriptRunner(
            script=os.path.join(HERE, "src", "client.py"),
            script_args=(f"--data_dir {data_dir} --metrics_dir {metrics_dir} --method {method} "
                         f"--epochs {args.epochs} --lr {args.lr} --batch_size {args.batch_size}"),
        )
        for site in sites:
            job.to(runner, site)
        simulator_run(job, workspace, sites, args.threads or len(sites), args.task_interval)

    results_dir = os.path.join(HERE, "results", method, dataset)
    os.makedirs(results_dir, exist_ok=True)
    metrics = pd.concat(pd.read_csv(f) for f in sorted(glob.glob(os.path.join(metrics_dir, "*.csv"))))
    curves = pd.concat(pd.read_csv(f) for f in sorted(glob.glob(os.path.join(metrics_dir, "curves", "*.csv"))))
    metrics.to_csv(os.path.join(results_dir, "metrics.csv"), index=False)
    curves.to_csv(os.path.join(results_dir, "curves.csv"), index=False)
    summarize(metrics, task, args.epochs)
    plots.plot_dataset(dataset, method, os.path.join(HERE, "results"), data_dir, task)


def summarize(df, task, epochs):
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", "{:.4f}".format)
    score = [m for m in task.metrics if m not in ("loss", "events")]
    extra = ["fs_r2"] if "fs_r2" in df.columns else []
    cols = ["site", "n_train", "n_test", *extra, *task.metrics]
    for rnd, g in df.groupby("round"):
        for stage, label in (("global", "global model received"),
                             ("local", f"after {epochs} local epochs")):
            sub = g[g.stage == stage].sort_values("site")
            print(f"\n=== Round {rnd}: {label} (per-client test split) ===")
            print(sub[cols].to_string(index=False))
            w = sub.n_test
            print("weighted mean: " + " ".join(f"{k}={(sub[k] * w).sum() / w.sum():.4f}" for k in extra + score))


def run_parallel(pairs, args):
    """Run each (dataset, method) pair as a subprocess of this script, args.jobs at a time."""
    log_dir = os.path.join(args.workspace, "logs")
    os.makedirs(log_dir, exist_ok=True)
    passthrough = [f"--rounds={args.rounds}", f"--epochs={args.epochs}", f"--lr={args.lr}",
                   f"--batch_size={args.batch_size}", f"--workspace={args.workspace}", f"--engine={args.engine}",
                   f"--task_interval={args.task_interval}"]
    if args.threads:
        passthrough.append(f"--threads={args.threads}")

    def one(pair):
        dataset, method = pair
        log = os.path.join(log_dir, f"{method}.{dataset}.log")
        with open(log, "w") as f:
            rc = subprocess.run([sys.executable, __file__, "--dataset", dataset, "--method", method, *passthrough],
                                stdout=f, stderr=subprocess.STDOUT).returncode
        return dataset, method, rc, log

    print(f"running {len(pairs)} jobs, {args.jobs} at a time; logs in {log_dir}", flush=True)
    failed = []
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        for fut in as_completed(ex.submit(one, pr) for pr in pairs):
            dataset, method, rc, log = fut.result()
            text = open(log).read()
            if rc != 0:
                failed.append((dataset, method))
                print(f"\n##### {dataset} / {method}: FAILED (exit {rc}), see {log}\n" + text[-2000:], flush=True)
                continue
            start = text.find("=== Round")
            print(f"\n##### {dataset} / {method}: done #####\n" + (text[start:] if start >= 0 else ""), flush=True)
    if failed:
        raise SystemExit(f"{len(failed)} job(s) failed: {failed}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", action="append", help="subfolder of data/simulated_data/federated/ (repeatable)")
    p.add_argument("--all", action="store_true", help="run every dataset under federated/")
    p.add_argument("--method", action="append", choices=["naive", "2sri", "2sps"],
                   help="repeatable; default 2sri")
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--batch_size", type=int, default=0, help="0 = task default (256; 512 for survival)")
    p.add_argument("--threads", type=int, default=None, help="simulator threads (default: one per site)")
    p.add_argument("--workspace", default=os.path.join(HERE, "workspace"))
    p.add_argument("--jobs", type=int, default=1, help="run this many (dataset, method) jobs concurrently")
    p.add_argument("--task_interval", type=float, default=0.05,
                   help="nvflare engine: seconds clients wait between task requests (NVFlare default 2)")
    p.add_argument("--engine", choices=["nvflare", "local"], default="nvflare",
                   help="nvflare simulator (real federation) or local in-process FedAvg (fast, same arithmetic)")
    args = p.parse_args()

    if args.all:
        datasets = sorted(d for d in os.listdir(FED_DIR) if os.path.isfile(os.path.join(FED_DIR, d, "manifest.json")))
    else:
        datasets = args.dataset or ["quadratic"]
    pairs = [(d, m) for m in (args.method or ["2sri"]) for d in datasets]
    if args.jobs > 1 and len(pairs) > 1:
        run_parallel(pairs, args)
    else:
        for d, m in pairs:
            run_dataset(d, m, args)


if __name__ == "__main__":
    main()
