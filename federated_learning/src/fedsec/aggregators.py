"""Server-side aggregation rules over the sites' updates (deltas from the global model).

Every rule takes U, a K x P array (one flattened update per site), and w, K
non-negative weights summing to 1, and returns the aggregate update (P,) and a
dict of per-site diagnostics used for the detection metrics:

    kept      share of coordinates at which the site's value entered the aggregate
    share     the site's effective weight in the aggregate relative to its nominal weight w
              (1 = used as FedAvg would; for the median, relative to the average site)
    flagged   clipped by norm bounding, or share < 0.5

Rules and the assumption each one needs to bound a malicious site's influence:

    fedavg            weighted mean. No robustness: one site can move the aggregate anywhere.
    median            coordinate-wise (weighted) median. Honest sites hold > half the weight.
    trimmed_mean      per coordinate drop the b = floor(trim_fraction * K) lowest and highest values,
                      weighted mean of the rest. At most b malicious sites.
    krum, multi_krum  Blanchard et al. 2017: score each update by the summed squared distance to its
                      K - f - 2 nearest others; krum keeps the best one, multi_krum the m best
                      (lowest scores, one pass) and averages them with their weights. K >= 2f + 3.
    geometric_median  RFA (Pillutla et al. 2022): weighted geometric median by smoothed Weiszfeld.
                      Breakdown point 1/2 of the weight.

All robust rules pay for robustness with efficiency: under heterogeneous but
honest sites they down-weight the sites that differ most, which here are the
small sites with noisy updates. Coordinate-wise rules also ignore the
correlation between coordinates of one network. The weighted rules keep the
sample-size weights (weights: reported/registered) unless weights: uniform.
"""

import numpy as np


def norm_bound(U, bound):
    """Scale every row of U to L2 norm <= bound. Returns (U_bounded, clipped mask, norms)."""
    norms = np.linalg.norm(U, axis=1)
    factor = np.minimum(1.0, bound / np.maximum(norms, 1e-12))
    return U * factor[:, None], norms > bound, norms


def fedavg(U, w, **_):
    K = len(w)
    return w @ U, {"kept": np.ones(K), "share": np.ones(K)}


def _weighted_median_cols(U, w):
    """Per column, the smallest value whose cumulative weight reaches half the total.
    Returns (median values, index of the chosen site per column)."""
    order = np.argsort(U, axis=0, kind="stable")
    cum = np.cumsum(w[order], axis=0)
    pos = np.argmax(cum >= 0.5 * w.sum() - 1e-12, axis=0)
    chosen = order[pos, np.arange(U.shape[1])]
    return U[chosen, np.arange(U.shape[1])], chosen


def median(U, w, **_):
    K, P = U.shape
    if np.allclose(w, w[0]):
        agg = np.median(U, axis=0)
        # a site "enters" the median if its value is one of the (one or two) middle order statistics
        ranks = np.argsort(np.argsort(U, axis=0, kind="stable"), axis=0, kind="stable")
        mid = {(K - 1) // 2, K // 2}
        used = np.isin(ranks, list(mid))
    else:
        agg, chosen = _weighted_median_cols(U, w)
        used = chosen[None, :] == np.arange(K)[:, None]
    kept = used.mean(axis=1)
    # every site is the middle value at ~1/K of the coordinates by chance; share is relative to that
    return agg, {"kept": kept, "share": kept / max(kept.mean(), 1e-12)}


def trimmed_mean(U, w, trim_fraction=0.1, **_):
    K, P = U.shape
    b = int(trim_fraction * K)
    order = np.argsort(U, axis=0, kind="stable")
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.arange(K)[:, None].repeat(P, axis=1), axis=0)
    keep = (ranks >= b) & (ranks < K - b)                  # K x P
    wk = keep * w[:, None]
    agg = (wk * U).sum(axis=0) / wk.sum(axis=0)
    kept = keep.mean(axis=1)
    # effective weight of site k averaged over coordinates, relative to its nominal weight
    share = (wk / wk.sum(axis=0, keepdims=True)).mean(axis=1) / np.maximum(w, 1e-12)
    return agg, {"kept": kept, "share": share}


def _krum_scores(U, f):
    K = len(U)
    d2 = ((U[:, None, :] - U[None, :, :]) ** 2).sum(-1)
    n_near = max(K - f - 2, 1)
    return np.array([np.sort(np.delete(d2[i], i))[:n_near].sum() for i in range(K)])


def krum(U, w, krum_f=1, multi_krum_m=None, multi=False, **_):
    K = len(w)
    scores = _krum_scores(U, krum_f)
    m = (multi_krum_m or K - krum_f) if multi else 1
    sel = np.zeros(K, dtype=bool)
    sel[np.argsort(scores, kind="stable")[:m]] = True
    ws = w * sel
    agg = (ws / ws.sum()) @ U
    return agg, {"kept": sel.astype(float), "share": np.where(sel, 1.0 / ws.sum(), 0.0), "krum_score": scores}


def multi_krum(U, w, **kw):
    return krum(U, w, multi=True, **kw)


def geometric_median(U, w, gm_max_iter=100, gm_tol=1e-7, gm_nu=1e-6, **_):
    """Weighted geometric median argmin_z sum_k w_k ||U_k - z|| by smoothed Weiszfeld (RFA)."""
    z = w @ U
    for _ in range(gm_max_iter):
        beta = w / np.maximum(np.linalg.norm(U - z, axis=1), gm_nu)
        z_new = (beta / beta.sum()) @ U
        if np.linalg.norm(z_new - z) <= gm_tol * max(np.linalg.norm(z), 1e-12):
            z = z_new
            break
        z = z_new
    beta = w / np.maximum(np.linalg.norm(U - z, axis=1), gm_nu)
    share = (beta / beta.sum()) / np.maximum(w, 1e-12)
    return z, {"kept": np.ones(len(w)), "share": share}


RULE_FNS = {"fedavg": fedavg, "median": median, "trimmed_mean": trimmed_mean, "krum": krum,
            "multi_krum": multi_krum, "geometric_median": geometric_median}


def aggregate(U, w, rule="fedavg", norm_bound_value=None, norm_bound_multiplier=1.0, **params):
    """Optional norm bounding, then the rule. Returns (aggregate, diagnostics incl. norms/clipped/bound)."""
    U = np.asarray(U, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    w = w / w.sum()
    norms = np.linalg.norm(U, axis=1)
    clipped = np.zeros(len(w), dtype=bool)
    bound = np.nan
    if norm_bound_value is not None:
        bound = norm_bound_multiplier * float(np.median(norms)) if norm_bound_value == "adaptive" \
            else float(norm_bound_value)
        U, clipped, _ = norm_bound(U, bound)
    agg, info = RULE_FNS[rule](U, w, **params)
    info.update({"norm": norms, "clipped": clipped, "bound": bound})
    # flagged: the defense clipped the site or cut its effective weight below half its nominal weight
    info["flagged"] = clipped | (info["share"] < 0.5)
    return agg, info
