"""NVFlare client script (Client API).

Each site loads its own CSV, splits it 80/20 (stratified, fixed seed), and in
every federated round:
  1. receives the global model and evaluates it on the local 20% test split,
  2. trains for `--epochs` local epochs on the 80% train split,
  3. evaluates the locally updated model on the same test split,
  4. sends the updated weights (+ metrics) back to the server.

Metrics for both evaluations are printed and appended to
<metrics_dir>/<site>.csv so job.py can print a per-round summary table.
"""

import argparse
import csv
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

import nvflare.client as flare
from model import MLP

METRIC_NAMES = ["loss", "accuracy", "precision", "recall", "f1", "auc"]


def load_site(data_dir: str, site: str, test_size: float, seed: int):
    df = pd.read_csv(os.path.join(data_dir, f"{site}.csv"))
    x = df[["X"]].to_numpy(dtype=np.float32)
    y = df["Y"].to_numpy()
    if not set(np.unique(y)) <= {0, 1}:
        # Continuous outcome (linear/quadratic/threshold sets): classify Y > 0.
        y = (y > 0).astype(int)
    y = y.astype(np.float32)
    x_tr, x_te, y_tr, y_te = train_test_split(
        x, y, test_size=test_size, random_state=seed, stratify=y
    )
    as_t = lambda a: torch.from_numpy(a)
    return as_t(x_tr), as_t(y_tr), as_t(x_te), as_t(y_te)


@torch.no_grad()
def evaluate(model, x, y, loss_fn):
    model.eval()
    logits = model(x)
    prob = torch.sigmoid(logits).numpy()
    pred = (prob >= 0.5).astype(int)
    y_np = y.numpy().astype(int)
    return {
        "loss": float(loss_fn(logits, y).item()),
        "accuracy": accuracy_score(y_np, pred),
        "precision": precision_score(y_np, pred, zero_division=0),
        "recall": recall_score(y_np, pred, zero_division=0),
        "f1": f1_score(y_np, pred, zero_division=0),
        "auc": roc_auc_score(y_np, prob) if len(np.unique(y_np)) > 1 else float("nan"),
    }


def train(model, x, y, loss_fn, epochs, lr, batch_size, seed):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    n = len(x)
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(x[idx]), y[idx])
            loss.backward()
            opt.step()


def fmt(m):
    return " ".join(f"{k}={v:.4f}" for k, v in m.items())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--metrics_dir", required=True)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    flare.init()
    site = flare.get_site_name()
    torch.manual_seed(args.seed)

    x_tr, y_tr, x_te, y_te = load_site(args.data_dir, site, args.test_size, args.seed)
    print(f"[{site}] train n={len(x_tr)} test n={len(x_te)} "
          f"test prevalence={y_te.mean():.3f}", flush=True)

    model = MLP()
    loss_fn = nn.BCEWithLogitsLoss()
    os.makedirs(args.metrics_dir, exist_ok=True)
    metrics_path = os.path.join(args.metrics_dir, f"{site}.csv")
    # The simulator may relaunch this script between rounds, so only write the
    # header when the file does not exist yet (job.py clears the workspace).
    if not os.path.exists(metrics_path):
        with open(metrics_path, "w", newline="") as f:
            csv.writer(f).writerow(["site", "round", "stage", "n_train", "n_test", *METRIC_NAMES])

    while flare.is_running():
        input_model = flare.receive()
        rnd = input_model.current_round
        model.load_state_dict(input_model.params)

        global_m = evaluate(model, x_te, y_te, loss_fn)
        print(f"[{site}] round {rnd} global-model test: {fmt(global_m)}", flush=True)

        train(model, x_tr, y_tr, loss_fn, args.epochs, args.lr, args.batch_size,
              seed=args.seed + rnd)
        local_m = evaluate(model, x_te, y_te, loss_fn)
        print(f"[{site}] round {rnd} local-model  test: {fmt(local_m)}", flush=True)

        with open(metrics_path, "a", newline="") as f:
            w = csv.writer(f)
            for stage, m in (("global", global_m), ("local", local_m)):
                w.writerow([site, rnd, stage, len(x_tr), len(x_te), *[m[k] for k in METRIC_NAMES]])

        flare.send(flare.FLModel(
            params=model.cpu().state_dict(),
            params_type="FULL",
            metrics={k: float(v) for k, v in global_m.items()},
            meta={"NUM_STEPS_CURRENT_ROUND": len(x_tr)},
        ))


if __name__ == "__main__":
    main()
