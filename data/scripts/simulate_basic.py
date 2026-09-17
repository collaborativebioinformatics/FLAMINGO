"""Basic MR dataset: independent SNP instruments, one exposure, one outcome, linear or non-linear link.

Model (see docs/mr-simulation-model.md):
    G_j ~ Binomial(2, p_j)
    U   ~ N(0, 1)
    X   = sum_j beta_j G_j + gamma_x U + e_x
    Y   = theta X + gamma_y U + e_y

No pleiotropy, no LD. Writes individual-level data and the true parameters.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl


def _draw_exposure(rng, n, n_snps, h2_x, gamma_x):
    """Draw SNPs, confounder U and exposure X. Shared by the linear and non-linear models."""
    maf = rng.uniform(0.05, 0.5, n_snps)
    G = rng.binomial(2, maf, size=(n, n_snps)).astype(np.int8)

    # Per-SNP effects scaled so the instruments explain h2_x of Var(X) = 1.
    beta = rng.normal(0.0, 1.0, n_snps)
    var_g = 2 * maf * (1 - maf)
    beta *= np.sqrt(h2_x / np.sum(beta**2 * var_g))
    gx = (G - 2 * maf) @ beta

    U = rng.normal(0.0, 1.0, n)
    e_x = rng.normal(0.0, np.sqrt(1.0 - h2_x - gamma_x**2), n)
    X = gx + gamma_x * U + e_x
    return G, maf, beta, U, X


def _frame(n, G, U, X, Y):
    snp_cols = {f"snp{j}": G[:, j] for j in range(G.shape[1])}
    return pl.DataFrame({"id": np.arange(n), **snp_cols, "U": U, "X": X, "Y": Y})


def simulate(n, n_snps, theta, h2_x, gamma_x, gamma_y, seed):
    """Linear model: Y = theta X + gamma_y U + e_y."""
    rng = np.random.default_rng(seed)
    G, maf, beta, U, X = _draw_exposure(rng, n, n_snps, h2_x, gamma_x)
    e_y = rng.normal(0.0, 1.0, n)
    Y = theta * X + gamma_y * U + e_y
    truth = {
        "model": "linear",
        "n": n, "n_snps": n_snps, "theta": theta, "h2_x": h2_x,
        "gamma_x": gamma_x, "gamma_y": gamma_y, "seed": seed,
        "maf": maf.tolist(), "beta": beta.tolist(),
    }
    return _frame(n, G, U, X, Y), truth


def causal_curve(shape, x, theta1, theta2):
    """f(x): the true causal effect of X on Y, minus confounding and noise.

    quadratic: theta1 * x + theta2 * x^2
    threshold: theta1 * x                  below theta2, flat above it
    """
    if shape == "quadratic":
        return theta1 * x + theta2 * x**2
    if shape == "threshold":
        return theta1 * np.minimum(x, theta2)
    raise ValueError(f"unknown shape {shape!r}; use 'quadratic' or 'threshold'")


def simulate_nonlinear(n, n_snps, shape, theta1, theta2, h2_x, gamma_x, gamma_y, seed):
    """Non-linear model: Y = f(X) + gamma_y U + e_y, with f from causal_curve().

    Same SNPs, confounder and exposure as simulate(); only the X -> Y link differs.
    The truth file records the population-average slope E[f'(X)], which is what a
    linear MR estimator targets when the true curve is not a line.
    """
    rng = np.random.default_rng(seed)
    G, maf, beta, U, X = _draw_exposure(rng, n, n_snps, h2_x, gamma_x)
    e_y = rng.normal(0.0, 1.0, n)
    Y = causal_curve(shape, X, theta1, theta2) + gamma_y * U + e_y

    if shape == "quadratic":
        avg_slope = theta1 + 2 * theta2 * X.mean()
    else:
        avg_slope = theta1 * np.mean(X < theta2)
    truth = {
        "model": shape, "theta1": theta1, "theta2": theta2, "avg_slope": float(avg_slope),
        "n": n, "n_snps": n_snps, "h2_x": h2_x,
        "gamma_x": gamma_x, "gamma_y": gamma_y, "seed": seed,
        "maf": maf.tolist(), "beta": beta.tolist(),
    }
    return _frame(n, G, U, X, Y), truth


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--n", type=int, default=10_000)
    p.add_argument("--n-snps", type=int, default=20)
    p.add_argument("--shape", choices=["linear", "quadratic", "threshold"], default="linear",
                   help="form of the X -> Y link")
    p.add_argument("--theta", type=float, default=0.3, help="linear slope (theta, or theta1 when non-linear)")
    p.add_argument("--theta2", type=float, default=0.0,
                   help="quadratic: coefficient on X^2; threshold: X value above which the effect stops")
    p.add_argument("--h2-x", type=float, default=0.10, help="Var(X) explained by SNPs")
    p.add_argument("--gamma-x", type=float, default=0.3, help="confounder effect on X")
    p.add_argument("--gamma-y", type=float, default=0.3, help="confounder effect on Y")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", type=Path, default=Path("data/basic"))
    a = p.parse_args()

    if a.shape == "linear":
        df, truth = simulate(a.n, a.n_snps, a.theta, a.h2_x, a.gamma_x, a.gamma_y, a.seed)
    else:
        df, truth = simulate_nonlinear(a.n, a.n_snps, a.shape, a.theta, a.theta2,
                                       a.h2_x, a.gamma_x, a.gamma_y, a.seed)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(a.out.with_suffix(".parquet"))
    a.out.with_suffix(".truth.json").write_text(json.dumps(truth, indent=1))
    print(f"wrote {a.out}.parquet ({df.height} rows, {df.width} cols) and {a.out}.truth.json")


if __name__ == "__main__":
    main()
