"""Site-level differential privacy for the update stream: clipping, Gaussian noise, accounting.

Unit of privacy. The protected unit is one site's entire contribution to a
round (its update), which also covers any individual inside that site.
Neighbouring federations differ by removing one site's data (add/remove
adjacency; a site without data sends a zero update). Changing rather than
removing a site's data doubles the sensitivity; privacy.json reports that
epsilon too (`*_replace`).

Mechanism. All K sites take part in every round (no subsampling, so no
amplification), and each of the T training rounds releases one Gaussian
mechanism. The composition of T Gaussian mechanisms with sensitivity/noise
ratio 1/z is exactly mu-Gaussian DP with mu = sqrt(T) / z (Dong, Roth & Su
2022), converted to (epsilon, delta) by the tight formula

    delta(eps) = Phi(-eps/mu + mu/2) - e^eps Phi(-eps/mu - mu/2).

Sensitivity of the aggregate. The server forms sum_k w_k clip(u_k) with fixed
public weights w_k (weights: registered or uniform; they are not renormalised
when a site is absent). Removing site k moves the sum by at most w_k C, so the
noise std is z C max_k w_k and every site gets at least the reported epsilon;
sites with smaller weights get more protection than reported.

What is not covered: the first stage (it never leaves the site), the per-site
test metrics that the NVFlare Client API can report (job.py suppresses them to
the server when dp or secagg is on), and the evaluation-only final round
(no update is released in it).
"""

import math

import numpy as np
from scipy.special import log_ndtr


def clip(u, clip_norm):
    """Scale u to L2 norm <= clip_norm. Returns (clipped, original norm)."""
    n = float(np.linalg.norm(u))
    return u * min(1.0, clip_norm / max(n, 1e-12)), n


def gdp_delta(eps, mu):
    """delta(eps) of a mu-GDP mechanism."""
    if mu <= 0:
        return 0.0
    a = log_ndtr(-eps / mu + mu / 2)
    b = eps + log_ndtr(-eps / mu - mu / 2)
    return float(math.exp(a) - math.exp(b))


def gdp_epsilon(mu, delta):
    """Smallest eps with delta(eps) <= delta for a mu-GDP mechanism; inf if mu is inf."""
    if not math.isfinite(mu):
        return math.inf
    if mu <= 0 or gdp_delta(0.0, mu) <= delta:
        return 0.0
    lo, hi = 0.0, 1.0
    while gdp_delta(hi, mu) > delta:
        hi *= 2
        if hi > 1e6:
            return math.inf
    for _ in range(200):
        mid = (lo + hi) / 2
        if gdp_delta(mid, mu) > delta:
            lo = mid
        else:
            hi = mid
    return hi


def noise_plan(dp, secagg_on, weights):
    """Per-site noise std for client-side noise and the server-side noise std.

    weights: {site: normalized public weight}.
    Returns dict with client_std {site: std on the *unweighted* update}, server_std, clip_at_client."""
    K = len(weights)
    C, z = dp.clip_norm, dp.noise_multiplier
    w_max = max(weights.values())
    plan = {"client_std": {s: 0.0 for s in weights}, "server_std": 0.0,
            "clip_at_client": dp.mode in ("local", "distributed") or secagg_on}
    if dp.mode == "central":
        plan["server_std"] = z * C * w_max
    elif dp.mode == "local":
        plan["client_std"] = {s: z * C for s in weights}
    elif dp.mode == "distributed":
        m = dp.min_honest or K
        # noise on the weighted contribution w_k u_k has std z C w_max / sqrt(m); on the unweighted
        # update that is divided by w_k, so that sum_k w_k (u_k + xi_k) carries the right total
        plan["client_std"] = {s: z * C * w_max / (math.sqrt(m) * weights[s]) for s in weights}
    return plan


def account(dp, secagg_on, rounds, weights, noisy_sites):
    """Epsilon against (a) anyone who sees the released global models ('aggregate') and
    (b) the server / anyone who sees each site's individual message ('server')."""
    if not dp.enabled:
        return {"dp": False, "epsilon_aggregate": math.inf, "epsilon_server": math.inf}
    K = len(weights)
    z, T, delta = dp.noise_multiplier, rounds, dp.delta
    w_max = max(weights.values())
    m = dp.min_honest or K
    h = sum(1 for s in weights if s in noisy_sites)
    if dp.mode == "central":
        mu_agg = math.sqrt(T) / z
        mu_srv = math.inf                               # the server sees raw (clipped) updates or their raw sum
    elif dp.mode == "local":
        mu_agg = mu_srv = math.sqrt(T) / z              # the aggregate is post-processing of noisy messages
    else:  # distributed
        # only the sites that actually add noise count: total variance scales with h / m
        mu_agg = math.sqrt(T) / (z * math.sqrt(h / m)) if h else math.inf
        if secagg_on:
            mu_srv = mu_agg
        else:
            # site k's own message: sensitivity C, noise std z C w_max / (sqrt(m) w_k); worst case w_k = w_max
            mu_srv = math.sqrt(T) * math.sqrt(m) / z
    out = {"dp": True, "mode": dp.mode, "secagg": secagg_on, "rounds": T, "clip_norm": dp.clip_norm,
           "noise_multiplier": z, "delta": delta, "max_weight": w_max, "min_honest": m,
           "noise_adding_sites": h, "sites": K, "mu_aggregate": mu_agg, "mu_server": mu_srv,
           "epsilon_aggregate": gdp_epsilon(mu_agg, delta), "epsilon_server": gdp_epsilon(mu_srv, delta),
           "epsilon_aggregate_replace": gdp_epsilon(2 * mu_agg, delta),
           "epsilon_server_replace": gdp_epsilon(2 * mu_srv, delta),
           "adjacency": "add/remove one site's data (sensitivity C, or w_max C for the weighted sum); "
                        "*_replace: change one site's data (sensitivity doubled)"}
    return out
