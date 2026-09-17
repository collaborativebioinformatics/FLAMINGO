"""Binary-outcome MR dataset: same exposure model as simulate_basic.py, but
Y is a 0/1 case status instead of continuous.

Model (see docs/mr-simulation-model.md):
    G_j ~ Binomial(2, p_j)
    U   ~ N(0, 1)
    X   = sum_j beta_j G_j + gamma_x U + e_x
    L   = f(X) + gamma_y U + e_y            (continuous liability)
    Y   = binarize(L)                        (see --link)

f is the same causal_curve() as simulate_basic.py (linear/quadratic/threshold),
so a binary outcome keeps the same shape optionality as the continuous one.
Reuses _draw_exposure/_frame/causal_curve from simulate_basic.py rather than
duplicating them.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from simulate_basic import _draw_exposure, _frame, causal_curve  # noqa: E402


def _expit(z):
    return 1.0 / (1.0 + np.exp(-z))


def _find_intercept_for_prevalence(z, prevalence, lo=-20.0, hi=20.0, tol=1e-6, max_iter=100):
    """Bisect for alpha such that mean(expit(alpha + z)) ~= prevalence."""
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        p = _expit(mid + z).mean()
        if abs(p - prevalence) < tol:
            break
        if p < prevalence:
            lo = mid
        else:
            hi = mid
    return mid


def binarize_liability(liability, prevalence, link, rng):
    """Turn a continuous liability into a 0/1 outcome at the target prevalence.

    'liability': deterministic threshold at the empirical quantile of the
        standardized liability corresponding to `prevalence` (the standard
        liability-threshold model from genetics).
    'logistic': solves for an intercept so mean(expit(alpha + z)) ~= prevalence,
        then samples Y ~ Bernoulli(expit(alpha + z)) - adds real sampling noise
        on top of the liability, unlike the deterministic threshold.

    Returns (Y, realized_prevalence, intercept_or_None).
    """
    z = (liability - liability.mean()) / liability.std()
    if link == "liability":
        cutoff = np.quantile(z, 1 - prevalence)
        Y = (z > cutoff).astype(np.int8)
        return Y, float(Y.mean()), None
    if link == "logistic":
        intercept = _find_intercept_for_prevalence(z, prevalence)
        p = _expit(intercept + z)
        Y = rng.binomial(1, p).astype(np.int8)
        return Y, float(Y.mean()), float(intercept)
    raise ValueError(f"unknown link {link!r}; use 'liability' or 'logistic'")


def simulate_binary(n, n_snps, shape, theta1, theta2, h2_x, gamma_x, gamma_y, seed,
                     link="logistic", prevalence=0.3):
    """Binary-outcome model: Y = binarize(f(X) + gamma_y U + e_y).

    shape: 'linear' (f(x) = theta1 x), 'quadratic', or 'threshold' (see
    causal_curve() in simulate_basic.py for the latter two).
    """
    rng = np.random.default_rng(seed)
    G, maf, beta, U, X = _draw_exposure(rng, n, n_snps, h2_x, gamma_x)
    signal = theta1 * X if shape == "linear" else causal_curve(shape, X, theta1, theta2)
    e_y = rng.normal(0.0, 1.0, n)
    liability = signal + gamma_y * U + e_y
    Y, realized_prevalence, intercept = binarize_liability(liability, prevalence, link, rng)

    if shape == "quadratic":
        avg_slope = theta1 + 2 * theta2 * X.mean()
    elif shape == "threshold":
        avg_slope = theta1 * np.mean(X < theta2)
    else:
        avg_slope = theta1

    truth = {
        "model": f"binary-{shape}", "theta1": theta1, "theta2": theta2, "avg_slope": float(avg_slope),
        "link": link, "prevalence_target": prevalence, "prevalence_realized": realized_prevalence,
        "intercept": intercept,
        "n": n, "n_snps": n_snps, "h2_x": h2_x, "gamma_x": gamma_x, "gamma_y": gamma_y, "seed": seed,
        "maf": maf.tolist(), "beta": beta.tolist(),
    }
    return _frame(n, G, U, X, Y), truth


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--n", type=int, default=10_000)
    p.add_argument("--n-snps", type=int, default=20)
    p.add_argument("--shape", choices=["linear", "quadratic", "threshold"], default="linear",
                   help="form of the X -> liability link")
    p.add_argument("--theta1", type=float, default=0.3, help="linear slope, or quadratic/threshold theta1")
    p.add_argument("--theta2", type=float, default=0.0, help="quadratic curvature, or threshold cutoff")
    p.add_argument("--h2-x", type=float, default=0.10)
    p.add_argument("--gamma-x", type=float, default=0.3)
    p.add_argument("--gamma-y", type=float, default=0.3)
    p.add_argument("--link", choices=["liability", "logistic"], default="logistic",
                   help="how the continuous liability is turned into 0/1")
    p.add_argument("--prevalence", type=float, default=0.3, help="target P(Y=1)")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", type=Path, default=Path("simulated_data/single/binary"))
    a = p.parse_args()

    df, truth = simulate_binary(a.n, a.n_snps, a.shape, a.theta1, a.theta2,
                                 a.h2_x, a.gamma_x, a.gamma_y, a.seed, a.link, a.prevalence)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(a.out.with_name(a.out.name + ".csv"))
    a.out.with_name(a.out.name + ".truth.json").write_text(json.dumps(truth, indent=1))
    print(f"wrote {a.out}.csv ({df.height} rows, {df.width} cols) and {a.out}.truth.json  "
          f"prevalence: target={a.prevalence:.3f} realized={truth['prevalence_realized']:.3f}")


if __name__ == "__main__":
    main()
