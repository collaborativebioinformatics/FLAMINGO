"""In-process FedAvg without NVFlare, for fast sweeps.

Runs the same Site code as the NVFlare client script (fedsite.py), the same
80/20 split, first stage, local epochs and evaluation, and the same
train-size-weighted averaging of weights that FedAvg does. What it does not
do is stand up a simulated server and ten client processes, which is where
the NVFlare simulator spends ~40 s per job regardless of model size.

Use engine="nvflare" (job.py default) to run the real federation; use
engine="local" to reproduce its arithmetic in a few seconds.
"""

import copy

import torch

from fedsite import Site, make_model


def fedavg(states, weights):
    total = float(sum(weights))
    avg = {}
    for k in states[0]:
        avg[k] = sum(s[k].float() * (w / total) for s, w in zip(states, weights)).to(states[0][k].dtype)
    return avg


def run(sites, data_dir, metrics_dir, method, rounds, epochs, lr, batch_size=0, test_size=0.2, seed=0, secure=None,
        init_seed=None, resample=None):
    """secure: a fedsec.config.SecureConfig. None, or one with every component off, runs plain FedAvg below.
    init_seed: seed for the initial weights (default: seed). resample: {site: SeedSequence entropy}
    for a bootstrap replicate (bootstrap.py); None leaves every site's data as it is."""
    torch.manual_seed(seed if init_seed is None else init_seed)
    global_params = copy.deepcopy(make_model(method).state_dict())
    clients = [Site(s, data_dir, metrics_dir, method, epochs, lr, batch_size, test_size, seed,
                    resample=(resample or {}).get(s)) for s in sites]
    for c in clients:
        print(c.describe(), flush=True)
    if secure is not None and secure.active:
        from fedsec.protocol import run_local
        return run_local(clients, global_params, rounds, secure, seed, metrics_dir)
    # Sites run one after another: the ops are too small for threads to help (the
    # GIL dominates), and parallelism across (dataset, method) jobs comes from --jobs.
    for rnd in range(rounds):
        states = [copy.deepcopy(c.run_round(rnd, global_params)[0]) for c in clients]
        global_params = fedavg(states, [c.n_train for c in clients])
    for c in clients:                       # evaluation-only pass over the final aggregate
        c.run_round(rounds, global_params, train=False)
    return global_params
