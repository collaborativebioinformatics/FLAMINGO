"""NVFlare client script (Client API), generic over outcome family and method.

Methods
  naive  out = f(X)
  2sri   out = f(X) + h(X - X_hat)     X_hat from a site-local OLS first stage X ~ SNPs,
  2sps   out = f(X_hat)                fitted once on the train split before federation

Each site loads its own CSV, detects the task from the columns (tasks.py),
splits 80/20 with a fixed seed, fits its first stage if the method needs one,
and every round: evaluates the received global model on its test split,
records the causal-curve head f on a fixed X grid, trains for `--epochs`
local epochs, evaluates again, and sends the weights back. The first-stage
R^2 on the test split (instrument strength) is reported as fs_r2.

Rows go to <metrics_dir>/<site>.csv and <metrics_dir>/curves/<site>.csv.
"""

import argparse
import csv
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

import nvflare.client as flare
from model import MLP, MRModel
from tasks import X_GRID, detect_task, load_manifest


def split(df, task, test_size, seed):
    tgt = task.targets(df)
    strat = tgt["strat"].numpy() if task.stratify else None
    idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=test_size,
                                      random_state=seed, stratify=strat)
    x = torch.tensor(df[["X"]].to_numpy(dtype=np.float32))
    g = torch.tensor(df[[c for c in df.columns if c.startswith("snp")]].to_numpy(dtype=np.float32))
    take = lambda idx: (x[idx], g[idx], {k: v[idx] for k, v in tgt.items() if k != "strat"})
    return take(idx_tr), take(idx_te)


def first_stage(g_tr, x_tr, g_te, x_te):
    """Site-local OLS of X on [1, SNPs]; returns X_hat for train and test and the test R^2."""
    z_tr = np.column_stack([np.ones(len(g_tr)), g_tr.numpy()])
    z_te = np.column_stack([np.ones(len(g_te)), g_te.numpy()])
    beta = np.linalg.lstsq(z_tr, x_tr.squeeze(-1).numpy(), rcond=None)[0]
    xhat_tr, xhat_te = z_tr @ beta, z_te @ beta
    r2 = r2_score(x_te.squeeze(-1).numpy(), xhat_te)
    as_t = lambda a: torch.tensor(a, dtype=torch.float32).unsqueeze(-1)
    return as_t(xhat_tr), as_t(xhat_te), float(r2)


@torch.no_grad()
def evaluate(model, task, x, xhat, tgt, fs_r2=None):
    model.eval()
    m = task.evaluate(model(x, xhat), tgt)
    if fs_r2 is not None:
        m["fs_r2"] = fs_r2
    return m


def train(model, task, x, xhat, tgt, epochs, lr, batch_size, seed):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed)
    n = len(x)
    for _ in range(epochs):
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            xh = None if xhat is None else xhat[idx]
            loss = task.loss(model(x[idx], xh), {k: v[idx] for k, v in tgt.items()})
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
    p.add_argument("--method", default="2sri", choices=["naive", "2sri", "2sps"])
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
    (x_tr, g_tr, t_tr), (x_te, g_te, t_te) = split(df, task, args.test_size, args.seed)
    if args.method == "naive":
        xhat_tr = xhat_te = fs_r2 = None
        model = MLP()
    else:
        xhat_tr, xhat_te, fs_r2 = first_stage(g_tr, x_tr, g_te, x_te)
        model = MRModel(args.method)
    print(f"[{site}] task={task.name} method={args.method} train n={len(x_tr)} test n={len(x_te)} "
          f"snps={g_tr.shape[1]}" + ("" if fs_r2 is None else f" first-stage test R2={fs_r2:.4f}"), flush=True)
    os.makedirs(os.path.join(args.metrics_dir, "curves"), exist_ok=True)
    metrics_path = os.path.join(args.metrics_dir, f"{site}.csv")
    curves_path = os.path.join(args.metrics_dir, "curves", f"{site}.csv")
    x_grid = torch.tensor(X_GRID).unsqueeze(1)
    cols = [*task.metrics, "fs_r2"] if args.method != "naive" else list(task.metrics)

    while flare.is_running():
        input_model = flare.receive()
        rnd = input_model.current_round
        model.load_state_dict(input_model.params)

        global_m = evaluate(model, task, x_te, xhat_te, t_te, fs_r2)
        print(f"[{site}] round {rnd} global-model test: {fmt(global_m)}", flush=True)
        with torch.no_grad():
            fx = model.curve(x_grid).numpy()
        append_rows(curves_path, ["site", "round", "x", "f"],
                    [[site, rnd, float(xg), float(fg)] for xg, fg in zip(X_GRID, fx)])

        train(model, task, x_tr, xhat_tr, t_tr, args.epochs, args.lr, batch_size, seed=args.seed + rnd)
        local_m = evaluate(model, task, x_te, xhat_te, t_te, fs_r2)
        print(f"[{site}] round {rnd} local-model  test: {fmt(local_m)}", flush=True)

        append_rows(metrics_path, ["site", "round", "stage", "n_train", "n_test", *cols],
                    [[site, rnd, stage, len(x_tr), len(x_te), *[m[k] for k in cols]]
                     for stage, m in (("global", global_m), ("local", local_m))])

        flare.send(flare.FLModel(
            params=model.cpu().state_dict(),
            params_type="FULL",
            metrics={k: float(v) for k, v in global_m.items()},
            meta={"NUM_STEPS_CURRENT_ROUND": len(x_tr)},
        ))


if __name__ == "__main__":
    main()
