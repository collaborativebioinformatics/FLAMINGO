"""Outcome-family definitions shared by the client, the job, and the plots.

The simulated datasets under data/simulated_data/federated/<name>/ come in
three families, told apart by their columns and manifest.json:

  continuous  Y real-valued (linear, quadratic, threshold)   -> regression, MSE
  binary      Y in {0, 1}   (binary_quadratic_logistic)      -> logistic, BCE
  survival    time + event  (cox, cox_rare)                  -> Cox partial likelihood

Every family uses the same MLP on the single feature X; only what its scalar
output means changes: predicted Y, logit of P(Y = 1), or log relative hazard.
"""

import json
import os

import numpy as np
import torch
from lifelines.utils import concordance_index
from sklearn.metrics import (
    accuracy_score, f1_score, mean_absolute_error, mean_squared_error,
    precision_score, r2_score, recall_score, roc_auc_score,
)

X_GRID = np.linspace(-3.0, 3.0, 61, dtype=np.float32)


def load_manifest(data_dir):
    with open(os.path.join(data_dir, "manifest.json")) as f:
        return json.load(f)


def true_curve(manifest, x):
    """f(X) the simulator used, on the scale the model output lives on
    (Y for continuous, log hazard for survival). None for binary, whose
    liability-scale curve is not on the logit scale the model fits."""
    shape, t1, t2 = manifest.get("shape"), manifest["theta1"], manifest.get("theta2")
    if manifest.get("outcome") == "binary":
        return None
    if shape == "quadratic":
        return t1 * x + t2 * x**2
    if shape == "threshold":
        return t1 * np.minimum(x, t2)
    return t1 * x  # linear and cox


def _cox_ph_loss(log_h, time, event):
    """Negative Cox partial log-likelihood (Breslow ties), averaged over events."""
    order = torch.argsort(time, descending=True)
    log_h, event = log_h[order], event[order]
    log_cum_risk = torch.logcumsumexp(log_h, dim=0)
    n_events = event.sum()
    if n_events == 0:
        return log_h.sum() * 0.0
    return -((log_h - log_cum_risk) * event).sum() / n_events


class Task:
    name = ""
    key_metric = ""           # what FedAvg tracks as "best" on the server
    metrics = ()              # column order in metrics CSVs and plots
    default_batch_size = 64
    stratify = False
    curve_label = ""

    def targets(self, df):            # -> dict of torch tensors, incl. 'strat' for splitting
        raise NotImplementedError

    def loss(self, out, tgt):
        raise NotImplementedError

    def evaluate(self, out, tgt):     # out: model output on the eval split
        raise NotImplementedError


class Continuous(Task):
    name = "continuous"
    key_metric = "r2"
    metrics = ("loss", "mse", "mae", "r2")
    curve_label = "predicted Y"

    def targets(self, df):
        y = torch.tensor(df["Y"].to_numpy(dtype=np.float32))
        return {"y": y}

    def loss(self, out, tgt):
        return torch.nn.functional.mse_loss(out, tgt["y"])

    def evaluate(self, out, tgt):
        y, p = tgt["y"].numpy(), out.numpy()
        return {"loss": float(mean_squared_error(y, p)), "mse": mean_squared_error(y, p),
                "mae": mean_absolute_error(y, p), "r2": r2_score(y, p)}


class Binary(Task):
    name = "binary"
    key_metric = "accuracy"
    metrics = ("loss", "accuracy", "precision", "recall", "f1", "auc")
    stratify = True
    curve_label = "predicted P(Y = 1)"

    def targets(self, df):
        y = torch.tensor(df["Y"].to_numpy(dtype=np.float32))
        return {"y": y, "strat": y}

    def loss(self, out, tgt):
        return torch.nn.functional.binary_cross_entropy_with_logits(out, tgt["y"])

    def evaluate(self, out, tgt):
        y = tgt["y"].numpy().astype(int)
        prob = torch.sigmoid(out).numpy()
        pred = (prob >= 0.5).astype(int)
        return {"loss": float(self.loss(out, tgt)),
                "accuracy": accuracy_score(y, pred),
                "precision": precision_score(y, pred, zero_division=0),
                "recall": recall_score(y, pred, zero_division=0),
                "f1": f1_score(y, pred, zero_division=0),
                "auc": roc_auc_score(y, prob) if len(np.unique(y)) > 1 else float("nan")}

    @staticmethod
    def curve_transform(out):
        return 1.0 / (1.0 + np.exp(-out))


class Survival(Task):
    name = "survival"
    key_metric = "c_index"
    metrics = ("loss", "c_index", "events")
    default_batch_size = 512   # the partial likelihood needs risk sets, so larger batches
    stratify = True            # keep the event fraction equal across the split
    curve_label = "log hazard ratio vs X = 0"

    def targets(self, df):
        return {"time": torch.tensor(df["time"].to_numpy(dtype=np.float32)),
                "event": torch.tensor(df["event"].to_numpy(dtype=np.float32)),
                "strat": torch.tensor(df["event"].to_numpy())}

    def loss(self, out, tgt):
        return _cox_ph_loss(out, tgt["time"], tgt["event"])

    def evaluate(self, out, tgt):
        t, e, h = tgt["time"].numpy(), tgt["event"].numpy(), out.numpy()
        return {"loss": float(self.loss(out, tgt)),
                "c_index": concordance_index(t, -h, e),
                "events": float(e.sum())}

    @staticmethod
    def curve_transform(out):
        return out - out[len(out) // 2]   # anchor at X = 0, like theta * X


def detect_task(df, manifest=None):
    cols = set(df.columns)
    if {"time", "event"} <= cols:
        return Survival()
    if "Y" not in cols:
        raise ValueError(f"no outcome column in {sorted(cols)}")
    if (manifest or {}).get("outcome") == "binary" or set(np.unique(df["Y"])) <= {0, 1}:
        return Binary()
    return Continuous()
