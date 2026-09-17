"""Build and run the NVFlare FedAvg job in the local simulator.

    uv run python job.py                     # 10 sites, 5 rounds, 2 local epochs
    uv run python job.py --dataset quadratic # continuous Y, classified as Y > 0
    uv run python job.py --help

One simulated client per site CSV in data/simulated_data/federated/<dataset>/.
After the run, prints a per-round table of every client's test metrics for
the global model it received and for the model it produced after local
training. Raw rows are in <workspace>/metrics/<site>.csv.
"""

import argparse
import glob
import os
import shutil

import pandas as pd
from nvflare.app_opt.pt.job_config.fed_avg import FedAvgJob
from nvflare.job_config.script_runner import ScriptRunner

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from model import MLP  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="binary_quadratic_logistic",
                   help="subfolder of data/simulated_data/federated/")
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--threads", type=int, default=None,
                   help="simulator threads (default: one per site)")
    p.add_argument("--workspace", default=os.path.join(HERE, "workspace"))
    args = p.parse_args()

    data_dir = os.path.join(REPO, "data", "simulated_data", "federated", args.dataset)
    sites = sorted(os.path.basename(f)[:-4] for f in glob.glob(os.path.join(data_dir, "site*.csv")))
    if not sites:
        raise SystemExit(f"no site*.csv files in {data_dir}")
    metrics_dir = os.path.join(args.workspace, "metrics")
    shutil.rmtree(args.workspace, ignore_errors=True)

    job = FedAvgJob(
        name=f"fedavg_{args.dataset}",
        n_clients=len(sites),
        num_rounds=args.rounds,
        initial_model=MLP(),
        key_metric="accuracy",
    )
    runner = ScriptRunner(
        script=os.path.join(HERE, "src", "client.py"),
        script_args=(f"--data_dir {data_dir} --metrics_dir {metrics_dir} "
                     f"--epochs {args.epochs} --lr {args.lr} --batch_size {args.batch_size}"),
    )
    for site in sites:
        job.to(runner, site)

    job.simulator_run(args.workspace, clients=sites, threads=args.threads or len(sites))
    summarize(metrics_dir, args.epochs)


def summarize(metrics_dir, epochs):
    files = sorted(glob.glob(os.path.join(metrics_dir, "*.csv")))
    if not files:
        print("no metrics written")
        return
    df = pd.concat(pd.read_csv(f) for f in files)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", "{:.4f}".format)
    cols = ["site", "n_train", "n_test", "loss", "accuracy", "precision", "recall", "f1", "auc"]
    for rnd, g in df.groupby("round"):
        for stage, label in (("global", "global model received"),
                             ("local", f"after {epochs} local epochs")):
            sub = g[g.stage == stage].sort_values("site")
            print(f"\n=== Round {rnd}: {label} (per-client test split) ===")
            print(sub[cols].to_string(index=False))
            w = sub.n_test
            print("weighted mean: " + " ".join(
                f"{k}={(sub[k] * w).sum() / w.sum():.4f}"
                for k in ["accuracy", "precision", "recall", "f1", "auc"]))


if __name__ == "__main__":
    main()
