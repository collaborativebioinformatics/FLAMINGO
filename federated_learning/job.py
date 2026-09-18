"""Build and run the NVFlare FedAvg job in the local simulator.

    uv run python job.py --dataset quadratic                 # 2SRI MR on one dataset
    uv run python job.py --all --method naive --method 2sri  # every dataset, two methods
    uv run python job.py --help

Methods (src/model.py):
  naive  fit outcome on X directly: the confounded association
  2sri   site-local OLS first stage X ~ SNPs, then federated outcome ~ f(X) + h(X - X_hat);
         f is the causal curve, h the control function (default)
  2sps   site-local first stage, then federated outcome ~ f(X_hat)
  fedmr  exact federated 2SLS from summed sufficient statistics (src/fedmr_engine.py):
         no training, two rounds, equals the pooled fit; continuous outcomes only
The first stage stays local because every simulated site has its own SNP
effects and allele frequencies, so there is no shared instrument to learn
(fedmr switches to its shared-instrument protocol when the manifest says
the SNPs are shared).

One simulated client per site CSV in data/simulated_data/federated/<dataset>/.
The outcome family is detected from the data (src/tasks.py). Each run prints
per-round tables of every client's test metrics and writes
results/<method>/<dataset>/{metrics,curves}.csv plus plots.

Robust / private federation (src/fedsec, ROBUST_PRIVATE.md), off unless a config turns it on:
    uv run python job.py --dataset quadratic --secure_config configs/secure/robust_median.yaml
    uv run python job.py --dataset quadratic --secure_config configs/secure/dp_distributed_secagg.yaml \\
        --secure_set dp.noise_multiplier=2.0
Secure runs write results/secure/<config name>/<method>/<dataset>/ and leave results/<method>/ alone.
"""

import argparse
import glob
import json
import math
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
import fedmr_engine  # noqa: E402
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


def load_secure(args):
    """The fedsec config from --secure_config / --secure_set; every component off when neither is given."""
    from fedsec.config import load_config
    return load_config(args.secure_config, args.secure_set)


def results_root(secure):
    if secure.active:
        return os.path.join(HERE, "results", "secure", secure.name)
    return os.path.join(HERE, "results")


def public_sizes(data_dir, sites, test_size=0.2):
    """Training-split sizes, as the server would hold them from registration. Same arithmetic as
    sklearn's train_test_split (n_test = ceil(test_size * n)), so they equal Site.n_train."""
    sizes = {}
    for s in sites:
        n = len(pd.read_csv(os.path.join(data_dir, f"{s}.csv"), usecols=["X"]))
        sizes[s] = n - math.ceil(test_size * n)
    return sizes


def secure_job(name, sites, initial, task, rounds, secure, data_dir, metrics_dir, seed, settings_path):
    """FedAvgJob's server side with the fedsec aggregator plugged into NVFlare's FedAvg controller.
    Writes the clients' settings to settings_path (outside the workspace, which the simulator
    clears on start). Returns (job, privacy report)."""
    from nvflare.app_common.workflows.fedavg import FedAvg
    from nvflare.app_opt.pt.job_config.base_fed_job import BaseFedJob
    from fedsec.config import validate
    from fedsec.nvflare_aggregator import FedSecAggregator
    from fedsec.protocol import ParamSpec, privacy_report

    validate(secure, sites)
    sizes = public_sizes(data_dir, sites)
    spec = ParamSpec(initial.state_dict()).to_list()
    # the server gets the config, the public sizes and the parameter layout; never the secagg session key
    aggregator = FedSecAggregator(config=secure.to_dict(), sites=sites, sizes=sizes, param_spec=spec, seed=seed,
                                  log_dir=os.path.join(metrics_dir, "fedsec"))
    job = BaseFedJob(initial_model=initial, name=name, key_metric=task.key_metric)
    job.to_server(FedAvg(num_clients=len(sites), num_rounds=rounds + 1, persistor_id=job.comp_ids["persistor_id"],
                         aggregator=aggregator))
    os.makedirs(os.path.dirname(settings_path), exist_ok=True)
    with open(settings_path, "w") as f:
        json.dump({"config": secure.to_dict(), "sites": sites, "sizes": sizes, "param_spec": spec}, f)
    return job, privacy_report(secure, sites, sizes, rounds, seed)


def collect_secure_logs(metrics_dir, results_dir):
    """fedsec logs from the workspace into the results folder: fedsec_{clients,server}.csv, privacy.json."""
    fs_dir = os.path.join(metrics_dir, "fedsec")
    if not os.path.isdir(fs_dir):
        return
    clients = sorted(glob.glob(os.path.join(fs_dir, "clients*.csv")))
    if clients:
        (pd.concat(pd.read_csv(f) for f in clients).sort_values(["round", "site"])
         .to_csv(os.path.join(results_dir, "fedsec_clients.csv"), index=False))
    for src, dst in (("server.csv", "fedsec_server.csv"), ("privacy.json", "privacy.json")):
        if os.path.isfile(os.path.join(fs_dir, src)):
            shutil.copyfile(os.path.join(fs_dir, src), os.path.join(results_dir, dst))


def summarize_secure(results_dir):
    path = os.path.join(results_dir, "privacy.json")
    if not os.path.isfile(path):
        return
    rep = json.load(open(path))
    print(f"\n=== fedsec: {rep['config']['name']} ===")
    print(f"malicious sites: {rep['malicious_sites'] or 'none'}   "
          f"epsilon (delta={rep['config']['dp']['delta']:g}): aggregate {rep['epsilon_aggregate']}, "
          f"server {rep['epsilon_server']}")
    sp = os.path.join(results_dir, "fedsec_server.csv")
    if os.path.isfile(sp):
        s = pd.read_csv(sp)
        for label, g in (("malicious", s[s.malicious == 1]), ("honest", s[s.malicious == 0])):
            if len(g):
                print(f"{label:9s} site-rounds flagged by the server: {g.flagged.mean():.2f} ({len(g)} site-rounds)")
    cp = os.path.join(results_dir, "fedsec_clients.csv")
    if os.path.isfile(cp):
        c = pd.read_csv(cp)
        h = c[c.malicious == 0]
        if h.update_cos.notna().any():
            print(f"honest sites: cosine(message seen by server, true update) {h.update_cos.mean():.3f}; "
                  f"membership-inference AUC {h.mia_auc.mean():.3f}")


def run_dataset(dataset, method, args):
    data_dir = os.path.join(FED_DIR, dataset)
    sites = sorted(os.path.basename(f)[:-4] for f in glob.glob(os.path.join(data_dir, "site*.csv")))
    if not sites:
        raise SystemExit(f"no site*.csv files in {data_dir}")
    head = pd.read_csv(os.path.join(data_dir, f"{sites[0]}.csv"), nrows=2000)
    manifest = load_manifest(data_dir)
    task = detect_task(head, manifest)
    secure = load_secure(args)
    if method == "fedmr":
        if secure.active:
            raise SystemExit("fedmr sends sufficient statistics, not model updates: the fedsec components "
                             "(robust aggregation, dp, secagg, attacks) do not apply. Run it without --secure_config.")
        reason = None
        if task.name != "continuous":
            reason = "FedMR is linear 2SLS on a continuous Y"
        else:
            reason = fedmr_engine.supported(manifest, args.fedmr_basis, args.fedmr_crossfit)
        if reason:
            print(f"\n##### {dataset} / fedmr: skipped, {reason} #####\n", flush=True)
            return
        print(f"\n##### {dataset} / fedmr ({args.engine}): task={task.name}, {len(sites)} sites, "
              f"basis={args.fedmr_basis}, crossfit={args.fedmr_crossfit} #####\n", flush=True)
        workspace = os.path.join(args.workspace, method, dataset)
        shutil.rmtree(workspace, ignore_errors=True)
        root = results_root(secure)
        fedmr_engine.run(dataset, data_dir, os.path.join(root, method, dataset), workspace, args.engine,
                         args.fedmr_basis, args.fedmr_crossfit, args.seed, simulator_run, args.task_interval)
        plots.plot_dataset(dataset, method, root, data_dir, task)
        return
    print(f"\n##### {dataset} / {method} ({args.engine}): task={task.name}, {len(sites)} sites, "
          f"{args.rounds} rounds x {args.epochs} local epochs"
          f"{f', secure config {secure.name}' if secure.active else ''} #####\n", flush=True)

    workspace = os.path.join(args.workspace, *(["secure", secure.name] if secure.active else []), method, dataset)
    metrics_dir = os.path.join(workspace, "metrics")
    shutil.rmtree(workspace, ignore_errors=True)

    if args.engine == "local":
        local_engine.run(sites, data_dir, metrics_dir, method, args.rounds, args.epochs, args.lr, args.batch_size,
                         seed=args.seed, secure=secure)
    else:
        initial = MLP() if method == "naive" else MRModel(method)
        # one extra round so every client evaluates the final aggregate (clients skip training in it)
        script_args = (f"--data_dir {data_dir} --metrics_dir {metrics_dir} --method {method} "
                       f"--rounds {args.rounds} --epochs {args.epochs} --lr {args.lr} "
                       f"--batch_size {args.batch_size}")
        if args.seed:
            script_args += f" --seed {args.seed}"
        report = None
        if secure.active:
            settings = workspace.rstrip(os.sep) + ".fedsec.json"
            job, report = secure_job(f"fedsec_{method}_{dataset}", sites, initial, task, args.rounds, secure,
                                     data_dir, metrics_dir, args.seed, settings)
            script_args += f" --fedsec {settings}"
        else:
            job = FedAvgJob(name=f"fedavg_{method}_{dataset}", n_clients=len(sites), num_rounds=args.rounds + 1,
                            initial_model=initial, key_metric=task.key_metric)
        runner = ScriptRunner(script=os.path.join(HERE, "src", "client.py"), script_args=script_args)
        for site in sites:
            job.to(runner, site)
        simulator_run(job, workspace, sites, args.threads or len(sites), args.task_interval)
        if report is not None:
            from fedsec.protocol import PRIVACY_FILE, write_privacy
            os.makedirs(os.path.join(metrics_dir, "fedsec"), exist_ok=True)
            write_privacy(os.path.join(metrics_dir, "fedsec", PRIVACY_FILE), report)

    root = results_root(secure)
    results_dir = os.path.join(root, method, dataset)
    os.makedirs(results_dir, exist_ok=True)
    metrics = pd.concat(pd.read_csv(f) for f in sorted(glob.glob(os.path.join(metrics_dir, "*.csv"))))
    curves = pd.concat(pd.read_csv(f) for f in sorted(glob.glob(os.path.join(metrics_dir, "curves", "*.csv"))))
    metrics.to_csv(os.path.join(results_dir, "metrics.csv"), index=False)
    curves.to_csv(os.path.join(results_dir, "curves.csv"), index=False)
    summarize(metrics, task, args.epochs)
    if secure.active:
        collect_secure_logs(metrics_dir, results_dir)
        summarize_secure(results_dir)
    plots.plot_dataset(dataset, method, root, data_dir, task)


def summarize(df, task, epochs):
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", "{:.4f}".format)
    score = [m for m in task.metrics if m not in ("loss", "events")]
    extra = ["fs_r2"] if "fs_r2" in df.columns else []
    cols = ["site", "n_train", "n_test", *extra, *task.metrics]
    last = df["round"].max()
    for rnd, g in df.groupby("round"):
        for stage, label in (("global", "global model received"),
                             ("local", f"after {epochs} local epochs")):
            sub = g[g.stage == stage].sort_values("site")
            if sub.empty:
                continue
            if rnd == last:
                label = f"final aggregate after {last} rounds"
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
                   f"--task_interval={args.task_interval}", f"--fedmr_basis={args.fedmr_basis}",
                   f"--fedmr_crossfit={args.fedmr_crossfit}"]
    if args.threads:
        passthrough.append(f"--threads={args.threads}")
    if args.seed:
        passthrough.append(f"--seed={args.seed}")
    if args.secure_config:
        passthrough.append(f"--secure_config={args.secure_config}")
    for item in args.secure_set or []:
        passthrough.append(f"--secure_set={item}")

    def one(pair):
        dataset, method = pair
        log = os.path.join(log_dir, f"{method}.{dataset}.log")
        with open(log, "w") as f:
            rc = subprocess.run([sys.executable, __file__, "--dataset", dataset, "--method", method, "--child",
                                 *passthrough],
                                stdout=f, stderr=subprocess.STDOUT).returncode
        return dataset, method, rc, log

    print(f"running {len(pairs)} jobs, {args.jobs} at a time; logs in {log_dir}", flush=True)
    failed = []
    # MR jobs overlay the naive curve of the same dataset, so all naive jobs finish first.
    batches = [[p for p in pairs if p[1] == "naive"], [p for p in pairs if p[1] != "naive"]]
    for batch in batches:
        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            for fut in as_completed(ex.submit(one, pr) for pr in batch):
                dataset, method, rc, log = fut.result()
                text = open(log).read()
                if rc != 0:
                    failed.append((dataset, method))
                    print(f"\n##### {dataset} / {method}: FAILED (exit {rc}), see {log}\n" + text[-2000:], flush=True)
                    continue
                start = max(text.find("=== Round"), text.find("FedMR fit"))
                print(f"\n##### {dataset} / {method}: done #####\n" + (text[start:] if start >= 0 else ""), flush=True)
    if failed:
        raise SystemExit(f"{len(failed)} job(s) failed: {failed}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", action="append", help="subfolder of data/simulated_data/federated/ (repeatable)")
    p.add_argument("--all", action="store_true", help="run every dataset under federated/")
    p.add_argument("--method", action="append", choices=["naive", "2sri", "2sps", "fedmr"],
                   help="repeatable; default 2sri")
    p.add_argument("--fedmr_basis", choices=["linear", "quadratic"], default="linear",
                   help="fedmr: structural basis, [X] or [X, X^2] (default linear)")
    p.add_argument("--fedmr_crossfit", type=int, default=0,
                   help="fedmr: k-fold cross-fitted instrument, site-local first stages only (default 0 = off)")
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--batch_size", type=int, default=0, help="0 = task default (256; 512 for survival)")
    p.add_argument("--threads", type=int, default=None, help="simulator threads (default: one per site)")
    p.add_argument("--workspace", default=os.path.join(HERE, "workspace"))
    p.add_argument("--jobs", type=int, default=1, help="run this many (dataset, method) jobs concurrently")
    p.add_argument("--child", action="store_true", help=argparse.SUPPRESS)   # set on --jobs subprocesses
    p.add_argument("--task_interval", type=float, default=0.05,
                   help="nvflare engine: seconds clients wait between task requests (NVFlare default 2)")
    p.add_argument("--engine", choices=["nvflare", "local"], default="nvflare",
                   help="nvflare simulator (real federation) or local in-process FedAvg (fast, same arithmetic)")
    p.add_argument("--seed", type=int, default=0,
                   help="split, initial weights, and fedsec randomness (dp noise, masks, attacks)")
    p.add_argument("--secure_config", default=None,
                   help="fedsec YAML (configs/secure/); robust aggregation, attacks, dp, secagg. Default: all off")
    p.add_argument("--secure_set", action="append", metavar="SECTION.KEY=VALUE",
                   help="override one fedsec config key, e.g. dp.noise_multiplier=2 (repeatable)")
    args = p.parse_args()
    secure = load_secure(args)                  # fail on a bad config before any job starts

    if args.all:
        datasets = sorted(d for d in os.listdir(FED_DIR) if os.path.isfile(os.path.join(FED_DIR, d, "manifest.json")))
    else:
        datasets = args.dataset or ["quadratic"]
    methods = args.method or ["2sri"]
    pairs = [(d, m) for m in (["naive"] if "naive" in methods else []) + [m for m in methods if m != "naive"]
             for d in datasets]
    if args.jobs > 1 and len(pairs) > 1:
        run_parallel(pairs, args)
    else:
        for d, m in pairs:
            run_dataset(d, m, args)
    if not args.child:
        root = results_root(secure)
        done = sorted({d for m in plots.METHODS if os.path.isdir(os.path.join(root, m))
                       for d in os.listdir(os.path.join(root, m))
                       if os.path.isfile(os.path.join(root, m, d, "metrics.csv"))})
        summary = plots.plot_overview(done, root, FED_DIR, os.path.join(root, "fitted_curves_all.png"))
        print(f"\n{os.path.relpath(root, HERE)}/summary.csv (last-round weighted test metrics, all runs so far):")
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
