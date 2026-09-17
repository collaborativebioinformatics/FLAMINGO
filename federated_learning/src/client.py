"""NVFlare client script (Client API), generic over the outcome family.

Each site loads its own CSV, works out the task from its columns (see
tasks.py), splits 80/20 with a fixed seed, and in every federated round:
  1. receives the global model and evaluates it on the local 20% test split,
  2. records the global model's fitted curve on a fixed X grid,
  3. trains for `--epochs` local epochs on the 80% train split,
  4. evaluates the locally updated model on the same test split,
  5. sends the updated weights (+ metrics) back to the server.

Rows go to <metrics_dir>/<site>.csv and <metrics_dir>/curves/<site>.csv so
job.py can build the per-round summary and the plots.
"""

import argparse
import csv
import os

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

import nvflare.client as flare
from model import MLP
from tasks import X_GRID, detect_task, load_manifest


def split(df, task, test_size, seed):
    tgt = task.targets(df)
    strat = tgt["strat"].numpy() if task.stratify else None
    idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=test_size,
                                      random_state=seed, stratify=strat)
    x = torch.tensor(df[["X"]].to_numpy(dtype=np.float32))
    take = lambda idx: (x[idx], {k: v[idx] for k, v in tgt.items() if k != "strat"})
    return take(idx_tr), take(idx_te)


@torch.no_grad()
def predict(model, x):
    model.eval()
    return model(x)


def train(model, x, tgt, task, epochs, lr, batch_size, seed):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    n = len(x)
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss = task.loss(model(x[idx]), {k: v[idx] for k, v in tgt.items()})
            loss.backward()
            opt.step()


def fmt(m):
    return " ".join(f"{k}={v:.4f}" for k, v in m.items())


def append_rows(path, header, rows):
    new = not os.path.exists(path)     # the simulator may relaunch this script
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--metrics_dir", required=True)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--batch_size", type=int, default=0, help="0 = task default")
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    flare.init()
    site = flare.get_site_name()
    torch.manual_seed(args.seed)

    df = pd.read_csv(os.path.join(args.data_dir, f"{site}.csv"))
    task = detect_task(df, load_manifest(args.data_dir))
    batch_size = args.batch_size or task.default_batch_size
    (x_tr, t_tr), (x_te, t_te) = split(df, task, args.test_size, args.seed)
    print(f"[{site}] task={task.name} train n={len(x_tr)} test n={len(x_te)}", flush=True)

    model = MLP()
    os.makedirs(os.path.join(args.metrics_dir, "curves"), exist_ok=True)
    metrics_path = os.path.join(args.metrics_dir, f"{site}.csv")
    curves_path = os.path.join(args.metrics_dir, "curves", f"{site}.csv")
    x_grid = torch.tensor(X_GRID).unsqueeze(1)

    while flare.is_running():
        input_model = flare.receive()
        rnd = input_model.current_round
        model.load_state_dict(input_model.params)

        global_m = task.evaluate(predict(model, x_te), t_te)
        print(f"[{site}] round {rnd} global-model test: {fmt(global_m)}", flush=True)
        fx = predict(model, x_grid).numpy()
        append_rows(curves_path, ["site", "round", "x", "f"],
                    [[site, rnd, float(xg), float(fg)] for xg, fg in zip(X_GRID, fx)])

        train(model, x_tr, t_tr, task, args.epochs, args.lr, batch_size, seed=args.seed + rnd)
        local_m = task.evaluate(predict(model, x_te), t_te)
        print(f"[{site}] round {rnd} local-model  test: {fmt(local_m)}", flush=True)

        append_rows(metrics_path, ["site", "round", "stage", "n_train", "n_test", *task.metrics],
                    [[site, rnd, stage, len(x_tr), len(x_te), *[m[k] for k in task.metrics]]
                     for stage, m in (("global", global_m), ("local", local_m))])

        flare.send(flare.FLModel(
            params=model.cpu().state_dict(),
            params_type="FULL",
            metrics={k: float(v) for k, v in global_m.items()},
            meta={"NUM_STEPS_CURRENT_ROUND": len(x_tr)},
        ))


if __name__ == "__main__":
    main()
