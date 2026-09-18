"""One site's data, model and per-round logic, shared by the NVFlare client
script (client.py) and the in-process local engine (local_engine.py) so both
run exactly the same arithmetic.

Methods
  naive  out = f(X)
  2sri   out = f(X) + c (X - X_hat)     X_hat from a site-local OLS first stage X ~ SNPs,
  2sps   out = f(X_hat)                 fitted once on the train split before federation
"""

import csv
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from model import MLP, MRModel
from tasks import X_GRID, detect_task, load_manifest

# A 1 -> 32 -> 32 -> 1 network on batches of 64 is far too small for intra-op
# parallelism: with torch's default thread pool one site's two epochs take
# 1.5 s, with one thread 0.2 s. Ten NVFlare clients in one process would also
# oversubscribe the cores.
torch.set_num_threads(1)


def make_model(method):
    return MLP() if method == "naive" else MRModel(method)


def append_rows(path, header, rows):
    new = not os.path.exists(path)     # the NVFlare simulator may relaunch the client script
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerows(rows)


def fmt(m):
    return " ".join(f"{k}={v:.4f}" for k, v in m.items())


class Site:
    def __init__(self, name, data_dir, metrics_dir, method, epochs, lr, batch_size=0, test_size=0.2, seed=0,
                 resample=None):
        """resample: None, or a numpy SeedSequence entropy list; then the training split is
        replaced by a bootstrap draw of its rows (with replacement, same size) before the
        first stage is fitted. Used by bootstrap.py; the test split is never resampled."""
        self.name, self.method, self.epochs, self.lr, self.seed = name, method, epochs, lr, seed
        df = pd.read_csv(os.path.join(data_dir, f"{name}.csv"))
        self.task = detect_task(df, load_manifest(data_dir))
        self.batch_size = batch_size or self.task.default_batch_size
        (self.x_tr, g_tr, self.t_tr), (self.x_te, g_te, self.t_te) = self._split(df, test_size, seed)
        if resample is not None:
            idx = torch.as_tensor(np.random.default_rng(resample).integers(0, len(self.x_tr), len(self.x_tr)))
            self.x_tr, g_tr, self.t_tr = self.x_tr[idx], g_tr[idx], {k: v[idx] for k, v in self.t_tr.items()}
        self.n_train, self.n_test = len(self.x_tr), len(self.x_te)
        if method == "naive":
            self.xhat_tr = self.xhat_te = self.fs_r2 = None
        else:
            self.xhat_tr, self.xhat_te, self.fs_r2 = self._first_stage(g_tr, self.x_tr, g_te, self.x_te)
            self.xhat_sd = float(self.xhat_tr.std())      # spread of the instrument; 2SPS is only identified within it
        self.model = make_model(method)
        self.cols = [*self.task.metrics, "fs_r2", "xhat_sd"] if method != "naive" else list(self.task.metrics)
        os.makedirs(os.path.join(metrics_dir, "curves"), exist_ok=True)
        self.metrics_path = os.path.join(metrics_dir, f"{name}.csv")
        self.curves_path = os.path.join(metrics_dir, "curves", f"{name}.csv")
        self.x_grid = torch.tensor(X_GRID).unsqueeze(1)

    def describe(self):
        s = f"[{self.name}] task={self.task.name} method={self.method} train n={self.n_train} test n={self.n_test}"
        return s if self.fs_r2 is None else s + f" first-stage test R2={self.fs_r2:.4f}"

    def _split(self, df, test_size, seed):
        tgt = self.task.targets(df)
        strat = tgt["strat"].numpy() if self.task.stratify else None
        idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=test_size, random_state=seed, stratify=strat)
        x = torch.tensor(df[["X"]].to_numpy(dtype=np.float32))
        g = torch.tensor(df[[c for c in df.columns if c.startswith("snp")]].to_numpy(dtype=np.float32))
        take = lambda idx: (x[idx], g[idx], {k: v[idx] for k, v in tgt.items() if k != "strat"})
        return take(idx_tr), take(idx_te)

    @staticmethod
    def _first_stage(g_tr, x_tr, g_te, x_te):
        """Site-local OLS of X on [1, SNPs]; X_hat for train and test, and the test R^2."""
        z_tr = np.column_stack([np.ones(len(g_tr)), g_tr.numpy()])
        z_te = np.column_stack([np.ones(len(g_te)), g_te.numpy()])
        beta = np.linalg.lstsq(z_tr, x_tr.squeeze(-1).numpy(), rcond=None)[0]
        xhat_tr, xhat_te = z_tr @ beta, z_te @ beta
        r2 = r2_score(x_te.squeeze(-1).numpy(), xhat_te)
        as_t = lambda a: torch.tensor(a, dtype=torch.float32).unsqueeze(-1)
        return as_t(xhat_tr), as_t(xhat_te), float(r2)

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        m = self.task.evaluate(self.model(self.x_te, self.xhat_te), self.t_te)
        if self.fs_r2 is not None:
            m["fs_r2"], m["xhat_sd"] = self.fs_r2, self.xhat_sd
        return m

    def train(self, rnd):
        self.model.train()
        # foreach=False: the default path lazily imports torch.distributed (1.7 s) on the first step
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, foreach=False)
        gen = torch.Generator().manual_seed(self.seed + rnd)
        n = self.n_train
        for _ in range(self.epochs):
            perm = torch.randperm(n, generator=gen)
            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                opt.zero_grad()
                xh = None if self.xhat_tr is None else self.xhat_tr[idx]
                loss = self.task.loss(self.model(self.x_tr[idx], xh), {k: v[idx] for k, v in self.t_tr.items()})
                loss.backward()
                opt.step()

    def run_round(self, rnd, global_params, train=True):
        """Load the global weights, evaluate, record the curve, then (unless train=False, the
        evaluation-only pass over the final aggregate) train locally and evaluate again.
        Returns (state_dict to send back, metrics of the received global model)."""
        self.model.load_state_dict(global_params)
        global_m = self.evaluate()
        print(f"[{self.name}] round {rnd} global-model test: {fmt(global_m)}", flush=True)
        with torch.no_grad():
            fx = self.model.curve(self.x_grid).numpy()
        append_rows(self.curves_path, ["site", "round", "x", "f"],
                    [[self.name, rnd, float(xg), float(fg)] for xg, fg in zip(X_GRID, fx)])
        header = ["site", "round", "stage", "n_train", "n_test", *self.cols]
        if not train:
            append_rows(self.metrics_path, header,
                        [[self.name, rnd, "global", self.n_train, self.n_test, *[global_m[k] for k in self.cols]]])
            return self.model.state_dict(), global_m

        self.train(rnd)
        local_m = self.evaluate()
        print(f"[{self.name}] round {rnd} local-model  test: {fmt(local_m)}", flush=True)
        append_rows(self.metrics_path, header,
                    [[self.name, rnd, stage, self.n_train, self.n_test, *[m[k] for k in self.cols]]
                     for stage, m in (("global", global_m), ("local", local_m))])
        return self.model.state_dict(), global_m
