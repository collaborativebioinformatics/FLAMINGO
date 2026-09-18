"""What an observer of one site's message learns about that site.

The observer is the aggregation server, or anyone who records the traffic
between a site and the server ("someone tracks our update data"). It knows the
global model it sent and sees the site's message. Two probes, both computed at
the site from quantities the site already holds, and logged locally only:

update_cos  cosine similarity between the observer's best reading of the
            message and the site's true update u = local - global. 1 means the
            observer recovers the site's local model exactly (plain FedAvg);
            ~0 means the message says nothing about u (secure aggregation).
            Under local DP it measures how much of u survives the noise.
mia_auc     membership inference: the observer rebuilds the site's local model
            as global + (message read as an update) and scores each record by
            loss under the global model minus loss under the rebuilt model
            (records the site trained on should fit the local model better).
            AUC of that score between the site's training records (members)
            and its held-out test records (non-members); 0.5 = no leakage.
            Defined for continuous and binary outcomes; the Cox partial
            likelihood has no per-record loss, so survival reports NaN, as does
            a masked (secagg) message, which cannot be read as an update.

The probes are lower bounds on leakage: a stronger attacker (several rounds,
auxiliary data, gradient inversion) may learn more.
"""

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0 or not np.isfinite(na) or not np.isfinite(nb):
        return float("nan")
    return float(a @ b / (na * nb))


def per_record_loss(task_name, out, tgt):
    if task_name == "continuous":
        return (out - tgt["y"]) ** 2
    if task_name == "binary":
        return torch.nn.functional.binary_cross_entropy_with_logits(out, tgt["y"], reduction="none")
    return None


@torch.no_grad()
def mia_auc(site, make_model, global_state, observed_state, max_members, rng):
    task_name = site.task.name
    if task_name not in ("continuous", "binary"):
        return float("nan")
    idx = np.arange(site.n_train)
    if len(idx) > max_members:
        idx = np.sort(rng.choice(idx, max_members, replace=False))
    idx = torch.as_tensor(idx)
    xm = site.x_tr[idx]
    xhm = None if site.xhat_tr is None else site.xhat_tr[idx]
    tm = {k: v[idx] for k, v in site.t_tr.items()}
    losses = []
    for state in (global_state, observed_state):
        model = make_model(site.method)
        model.load_state_dict(state)
        model.eval()
        lm = per_record_loss(task_name, model(xm, xhm), tm)
        ln = per_record_loss(task_name, model(site.x_te, site.xhat_te), site.t_te)
        losses.append((lm.numpy(), ln.numpy()))
    score_m = losses[0][0] - losses[1][0]
    score_n = losses[0][1] - losses[1][1]
    scores = np.concatenate([score_m, score_n])
    if not np.all(np.isfinite(scores)):
        return float("nan")
    labels = np.concatenate([np.ones(len(score_m)), np.zeros(len(score_n))])
    return float(roc_auc_score(labels, scores))
