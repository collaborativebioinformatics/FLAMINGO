"""Bootstrap confidence bands for the FedAvg curves (Fed-2SRI, and naive / Fed-2SPS).

A FedAvg fit is a neural network trained on top of a site-local first stage,
so it has no closed-form standard error; even for classic 2SRI the naive
second-stage SE ignores the first stage. The interval therefore comes from
bootstrapping the whole procedure. Replicate b:

  1. every site draws its training rows with replacement (same n, test split untouched),
  2. refits its local first stage X ~ SNPs on that draw,
  3. the federation retrains from fresh initial weights with the same rounds, epochs and lr,

and the final global model's f(x) - f(0) is recorded on the X grid. The band
is the pointwise percentile interval of those curves. Nothing extra leaves a
site: a replicate is one more federated run, and the band is computed from
the global models only.

What the band means, and what it assumes:
- Individuals are independent within a site and sites are independent; the
  resampling is stratified by site and the set of sites is held fixed, so the
  band is conditional on these sites (sites are not resampled).
- It covers sampling variability of both stages plus the randomness of
  training (initial weights). It does not cover bias: if the procedure is
  biased for the true curve (few rounds, weak instruments, the 2SRI
  approximation on the logit / log-hazard scale), the band can miss the truth.
- It is pointwise, not simultaneous: each X has 95% coverage on its own.
- Replicates always run on the local engine (the same arithmetic as NVFlare,
  seconds instead of ~40 s per federated run); with --engine nvflare the point
  estimate comes from NVFlare and the band from local-engine replicates.

Outputs, next to the run's curves.csv:
  curves_bootstrap.csv   rep, x, f       one anchored final curve per replicate
  curves.csv             gains f_lo, f_hi on the last round, which plots.last_band draws
  bootstrap.json         B, level, seeds, and the assumptions above in brief
"""

import contextlib
import io
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch

from tasks import X_GRID

STREAM = 11   # SeedSequence stream id for the resampling draws


def _replicate(job):
    """One bootstrap replicate in a worker process. Returns the anchored final curve on X_GRID."""
    import local_engine
    from fedsite import make_model

    b, sites, data_dir, work, method, rounds, epochs, lr, batch_size, seed = job
    resample = {s: [seed, STREAM, b, k] for k, s in enumerate(sites)}
    metrics_dir = os.path.join(work, f"rep{b:04d}")
    with contextlib.redirect_stdout(io.StringIO()):
        params = local_engine.run(sites, data_dir, metrics_dir, method, rounds, epochs, lr, batch_size, seed=seed,
                                  init_seed=seed * 100_003 + 1 + b, resample=resample)
    shutil.rmtree(metrics_dir, ignore_errors=True)
    model = make_model(method)
    model.load_state_dict(params)
    model.eval()
    with torch.no_grad():
        f = model.curve(torch.tensor(X_GRID).unsqueeze(1)).numpy().astype(float)
    return b, f - np.interp(0.0, X_GRID, f)


def run(sites, data_dir, work, method, rounds, epochs, lr, batch_size=0, seed=0, n_boot=200, jobs=None):
    """B replicates, `jobs` at a time. Returns a long DataFrame rep, x, f."""
    jobs = jobs or max(1, (os.cpu_count() or 2) - 1)
    os.makedirs(work, exist_ok=True)
    tasks = [(b, list(sites), data_dir, work, method, rounds, epochs, lr, batch_size, seed) for b in range(n_boot)]
    # spawn, not fork: the parent has already run torch, and a forked OpenMP runtime can hang
    import multiprocessing as mp
    with ProcessPoolExecutor(max_workers=min(jobs, n_boot), mp_context=mp.get_context("spawn")) as ex:
        curves = dict(ex.map(_replicate, tasks, chunksize=max(1, n_boot // (4 * jobs))))
    shutil.rmtree(work, ignore_errors=True)
    return pd.DataFrame([(b, float(x), float(v)) for b in range(n_boot) for x, v in zip(X_GRID, curves[b])],
                        columns=["rep", "x", "f"])


def band(boot, level=0.95):
    """Pointwise percentile band of anchored curves: DataFrame x, lo, hi, sd."""
    a = (1 - level) / 2
    g = boot.groupby("x").f
    return pd.DataFrame({"lo": g.quantile(a), "hi": g.quantile(1 - a), "sd": g.std()}).reset_index()


def attach_band(curves, bnd):
    """curves.csv with f_lo / f_hi on its last round, so plots.last_band draws the band.

    last_band re-anchors at the point estimate's own f(0), so the anchored band is
    shifted by that f(0) here and comes back out exactly."""
    curves = curves.drop(columns=["f_lo", "f_hi"], errors="ignore").copy()
    last = curves["round"] == curves["round"].max()
    point = curves[last].groupby("x").f.mean().sort_index()
    f0 = float(np.interp(0.0, point.index.to_numpy(), point.to_numpy()))
    b = bnd.set_index("x")
    xs = curves.loc[last, "x"]
    # the grid is the same float32 X_GRID everywhere; match on the rounded value
    key = lambda s: np.round(np.asarray(s, dtype=float), 6)
    lo = pd.Series(b.lo.to_numpy(), index=key(b.index))
    hi = pd.Series(b.hi.to_numpy(), index=key(b.index))
    curves["f_lo"] = np.nan
    curves["f_hi"] = np.nan
    curves.loc[last, "f_lo"] = lo.reindex(key(xs)).to_numpy() + f0
    curves.loc[last, "f_hi"] = hi.reindex(key(xs)).to_numpy() + f0
    return curves


def write(results_dir, boot, level, meta):
    """curves_bootstrap.csv, band columns on curves.csv, bootstrap.json. Returns the band."""
    boot.to_csv(os.path.join(results_dir, "curves_bootstrap.csv"), index=False)
    bnd = band(boot, level)
    path = os.path.join(results_dir, "curves.csv")
    attach_band(pd.read_csv(path), bnd).to_csv(path, index=False)
    info = dict(meta, level=level, replicates=int(boot.rep.nunique()),
                interval="pointwise percentile of f(x) - f(0) over replicates",
                resampling="rows of each site's training split, with replacement; sites held fixed",
                refit="site-local first stage and the full FedAvg training, fresh initial weights per replicate",
                engine_for_replicates="local")
    with open(os.path.join(results_dir, "bootstrap.json"), "w") as f:
        json.dump(info, f, indent=1)
    return bnd
