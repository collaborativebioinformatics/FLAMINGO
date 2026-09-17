"""Build and run the NVFlare FedAvg job in the local simulator.

    uv run python job.py --dataset quadratic     # one dataset
    uv run python job.py --all                   # every dataset under federated/
    uv run python job.py --help

One simulated client per site CSV in data/simulated_data/federated/<dataset>/.
The outcome family (continuous / binary / survival) is detected from the
data, see src/tasks.py. After each run, prints a per-round table of every
client's test metrics, writes results/<dataset>/metrics.csv and curves.csv,
and renders the plots in results/<dataset>/.
"""

import argparse
import glob
import os
import shutil
import sys

import pandas as pd
from nvflare.app_opt.pt.job_config.fed_avg import FedAvgJob
from nvflare.job_config.script_runner import ScriptRunner

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
FED_DIR = os.path.join(REPO, "data", "simulated_data", "federated")
sys.path.insert(0, os.path.join(HERE, "src"))
from model import MLP  # noqa: E402
from tasks import detect_task, load_manifest  # noqa: E402
import plots  # noqa: E402


def run_dataset(dataset, args):
    data_dir = os.path.join(FED_DIR, dataset)
    sites = sorted(os.path.basename(f)[:-4] for f in glob.glob(os.path.join(data_dir, "site*.csv")))
    if not sites:
        raise SystemExit(f"no site*.csv files in {data_dir}")
    task = detect_task(pd.read_csv(os.path.join(data_dir, f"{sites[0]}.csv"), nrows=2000),
                       load_manifest(data_dir))
    print(f"\n##### {dataset}: task={task.name}, {len(sites)} sites, "
          f"{args.rounds} rounds x {args.epochs} local epochs #####\n", flush=True)

    workspace = os.path.join(args.workspace, dataset)
    metrics_dir = os.path.join(workspace, "metrics")
    shutil.rmtree(workspace, ignore_errors=True)

    job = FedAvgJob(name=f"fedavg_{dataset}", n_clients=len(sites), num_rounds=args.rounds,
                    initial_model=MLP(), key_metric=task.key_metric)
    runner = ScriptRunner(
        script=os.path.join(HERE, "src", "client.py"),
        script_args=(f"--data_dir {data_dir} --metrics_dir {metrics_dir} "
                     f"--epochs {args.epochs} --lr {args.lr} --batch_size {args.batch_size}"),
    )
    for site in sites:
        job.to(runner, site)
    job.simulator_run(workspace, clients=sites, threads=args.threads or len(sites))

    results_dir = os.path.join(HERE, "results", dataset)
    os.makedirs(results_dir, exist_ok=True)
    metrics = pd.concat(pd.read_csv(f) for f in sorted(glob.glob(os.path.join(metrics_dir, "*.csv"))))
    curves = pd.concat(pd.read_csv(f) for f in sorted(glob.glob(os.path.join(metrics_dir, "curves", "*.csv"))))
    metrics.to_csv(os.path.join(results_dir, "metrics.csv"), index=False)
    curves.to_csv(os.path.join(results_dir, "curves.csv"), index=False)
    summarize(metrics, task, args.epochs)
    plots.plot_dataset(dataset, results_dir, data_dir, task)


def summarize(df, task, epochs):
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", "{:.4f}".format)
    cols = ["site", "n_train", "n_test", *task.metrics]
    score = [m for m in task.metrics if m not in ("loss", "events")]
    for rnd, g in df.groupby("round"):
        for stage, label in (("global", "global model received"),
                             ("local", f"after {epochs} local epochs")):
            sub = g[g.stage == stage].sort_values("site")
            print(f"\n=== Round {rnd}: {label} (per-client test split) ===")
            print(sub[cols].to_string(index=False))
            w = sub.n_test
            print("weighted mean: " + " ".join(f"{k}={(sub[k] * w).sum() / w.sum():.4f}" for k in score))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", action="append", help="subfolder of data/simulated_data/federated/ (repeatable)")
    p.add_argument("--all", action="store_true", help="run every dataset under federated/")
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--batch_size", type=int, default=0, help="0 = task default (64; 512 for survival)")
    p.add_argument("--threads", type=int, default=None, help="simulator threads (default: one per site)")
    p.add_argument("--workspace", default=os.path.join(HERE, "workspace"))
    args = p.parse_args()

    if args.all:
        datasets = sorted(d for d in os.listdir(FED_DIR) if os.path.isfile(os.path.join(FED_DIR, d, "manifest.json")))
    else:
        datasets = args.dataset or ["binary_quadratic_logistic"]
    for d in datasets:
        run_dataset(d, args)


if __name__ == "__main__":
    main()
