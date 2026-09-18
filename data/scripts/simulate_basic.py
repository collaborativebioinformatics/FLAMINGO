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


def _draw_exposure(rng, n, n_snps, h2_x, gamma_x, maf=None, beta=None) -> tuple:
    """Draw SNPs, confounder U and exposure X. Shared by the linear and non-linear models.

    maf and beta default to fresh draws (site-specific variants). Pass both to
    reuse harmonized SNPs across sites: beta is then used as given, and h2_x is
    replaced by the variance those effects explain at these allele frequencies.
    Returns the h2_x actually used as the last element.
    """
    if maf is None:
        maf = rng.uniform(0.05, 0.5, n_snps)
    maf = np.asarray(maf, dtype=float)
    G = rng.binomial(2, maf, size=(n, n_snps)).astype(np.int8)

    var_g = 2 * maf * (1 - maf)
    if beta is None:
        # Per-SNP effects scaled so the instruments explain h2_x of Var(X) = 1.
        beta = rng.normal(0.0, 1.0, n_snps)
        beta *= np.sqrt(h2_x / np.sum(beta**2 * var_g))
    else:
        beta = np.asarray(beta, dtype=float)
        h2_x = float(np.sum(beta**2 * var_g))
    gx = (G - 2 * maf) @ beta

    U = rng.normal(0.0, 1.0, n)
    resid_var = 1.0 - h2_x - gamma_x**2
    if resid_var <= 0:
        raise ValueError(f"h2_x + gamma_x^2 must be < 1 so Var(X) = 1 is attainable; got {h2_x} + {gamma_x}^2")
    e_x = rng.normal(0.0, np.sqrt(resid_var), n)
    X = gx + gamma_x * U + e_x
    return G, maf, beta, U, X, h2_x


def scaled_beta(rng, maf, h2_x) -> np.ndarray:
    """Per-SNP effects explaining h2_x of Var(X) = 1 at allele frequencies maf (for shared SNPs)."""
    beta = rng.normal(0.0, 1.0, len(maf))
    return beta * np.sqrt(h2_x / np.sum(beta**2 * 2 * maf * (1 - maf)))


def _pleiotropy(G, maf, alpha) -> np.ndarray | float:
    """Direct G -> Y effects (horizontal pleiotropy), centred so they do not shift E[Y]."""
    if alpha is None:
        return 0.0
    return (G - 2 * maf) @ np.asarray(alpha, dtype=float)


def _frame(n, G, U, X, Y) -> pl.DataFrame:
    snp_cols = {f"snp{j}": G[:, j] for j in range(G.shape[1])}
    return pl.DataFrame({"id": np.arange(n), **snp_cols, "U": U, "X": X, "Y": Y})


def simulate(n, n_snps, theta, h2_x, gamma_x, gamma_y, seed, maf=None, beta=None, alpha=None) -> tuple:
    """Linear model: Y = theta X + sum_j alpha_j G_j + gamma_y U + e_y (alpha = 0 unless given)."""
    rng = np.random.default_rng(seed)
    G, maf, beta, U, X, h2_x = _draw_exposure(rng, n, n_snps, h2_x, gamma_x, maf, beta)
    e_y = rng.normal(0.0, 1.0, n)
    Y = theta * X + _pleiotropy(G, maf, alpha) + gamma_y * U + e_y
    truth = {
        "model": "linear",
        "n": n, "n_snps": n_snps, "theta": theta, "h2_x": h2_x,
        "gamma_x": gamma_x, "gamma_y": gamma_y, "seed": seed,
        "maf": maf.tolist(), "beta": beta.tolist(),
        **({"alpha": list(map(float, alpha))} if alpha is not None else {}),
    }
    return _frame(n, G, U, X, Y), truth


def causal_curve(shape, x, theta1, theta2) -> np.ndarray:
    """f(x): the true causal effect of X on Y, minus confounding and noise.

    quadratic: theta1 * x + theta2 * x^2
    threshold: theta1 * x                  below theta2, flat above it
    """
    if shape == "quadratic":
        return theta1 * x + theta2 * x**2
    if shape == "threshold":
        return theta1 * np.minimum(x, theta2)
    raise ValueError(f"unknown shape {shape!r}; use 'quadratic' or 'threshold'")


def simulate_nonlinear(n, n_snps, shape, theta1, theta2, h2_x, gamma_x, gamma_y, seed,
                       maf=None, beta=None, alpha=None) -> tuple:
    """Non-linear model: Y = f(X) + gamma_y U + e_y, with f from causal_curve().

    Same SNPs, confounder and exposure as simulate(); only the X -> Y link differs.
    The truth file records the population-average slope E[f'(X)], which is what a
    linear MR estimator targets when the true curve is not a line.
    """
    rng = np.random.default_rng(seed)
    G, maf, beta, U, X, h2_x = _draw_exposure(rng, n, n_snps, h2_x, gamma_x, maf, beta)
    e_y = rng.normal(0.0, 1.0, n)
    Y = causal_curve(shape, X, theta1, theta2) + _pleiotropy(G, maf, alpha) + gamma_y * U + e_y

    if shape == "quadratic":
        avg_slope = theta1 + 2 * theta2 * X.mean()
    else:
        avg_slope = theta1 * np.mean(X < theta2)
    truth = {
        "model": shape, "theta1": theta1, "theta2": theta2, "avg_slope": float(avg_slope),
        "n": n, "n_snps": n_snps, "h2_x": h2_x,
        "gamma_x": gamma_x, "gamma_y": gamma_y, "seed": seed,
        "maf": maf.tolist(), "beta": beta.tolist(),
        **({"alpha": list(map(float, alpha))} if alpha is not None else {}),
    }
    return _frame(n, G, U, X, Y), truth


def simulate_survival(n, n_snps, theta, h2_x, gamma_x, gamma_y, seed,
                      weibull_k=1.5, weibull_scale=10.0, censor_frac=0.3, followup=15.0) -> tuple:
    """Cox proportional-hazards outcome: h(t) = h0(t) exp(theta X + gamma_y U).

    h0 is Weibull with shape weibull_k and scale weibull_scale, so event times are
    T = scale * (-log(V) / exp(lp))^(1/k) with V ~ Uniform(0, 1). theta is the log
    hazard ratio per unit X. Random censoring is exponential, tuned so about
    censor_frac of subjects are censored before administrative end of follow-up.
    Columns `time` and `event` replace `Y`.
    """
    rng = np.random.default_rng(seed)
    G, maf, beta, U, X, h2_x = _draw_exposure(rng, n, n_snps, h2_x, gamma_x)
    lp = theta * X + gamma_y * U
    T = weibull_scale * (-np.log(rng.uniform(size=n)) / np.exp(lp)) ** (1.0 / weibull_k)

    # exponential censoring rate chosen so P(C < T) ~ censor_frac, then cap at follow-up
    median_T = np.median(T)
    rate = -np.log(1.0 - censor_frac) / median_T if censor_frac > 0 else 0.0
    C = rng.exponential(1.0 / rate, n) if rate > 0 else np.full(n, np.inf)
    C = np.minimum(C, followup)
    time = np.minimum(T, C)
    event = (T <= C).astype(np.int8)

    snp_cols = {f"snp{j}": G[:, j] for j in range(n_snps)}
    df = pl.DataFrame({"id": np.arange(n), **snp_cols, "U": U, "X": X, "time": time, "event": event})
    truth = {
        "model": "cox", "theta": theta, "hazard_ratio": float(np.exp(theta)),
        "weibull_k": weibull_k, "weibull_scale": weibull_scale,
        "censor_frac": censor_frac, "followup": followup, "event_rate": float(event.mean()),
        "n": n, "n_snps": n_snps, "h2_x": h2_x,
        "gamma_x": gamma_x, "gamma_y": gamma_y, "seed": seed,
        "maf": maf.tolist(), "beta": beta.tolist(),
    }
    return df, truth


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--n", type=int, default=10_000)
    p.add_argument("--n-snps", type=int, default=20)
    p.add_argument("--shape", choices=["linear", "quadratic", "threshold", "cox"], default="linear",
                   help="form of the X -> Y link; cox gives a survival outcome (time, event)")
    p.add_argument("--theta", type=float, default=0.3, help="linear slope (theta, or theta1 when non-linear; log hazard ratio for cox)")
    p.add_argument("--theta2", type=float, default=0.0,
                   help="quadratic: coefficient on X^2; threshold: X value above which the effect stops")
    p.add_argument("--h2-x", type=float, default=0.10, help="Var(X) explained by SNPs")
    p.add_argument("--gamma-x", type=float, default=0.3, help="confounder effect on X")
    p.add_argument("--gamma-y", type=float, default=0.3, help="confounder effect on Y")
    p.add_argument("--censor-frac", type=float, default=0.3, help="cox: target fraction randomly censored")
    p.add_argument("--followup", type=float, default=15.0, help="cox: administrative end of follow-up")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", type=Path, default=Path("simulated_data/single/basic"))
    a = p.parse_args()

    if a.shape == "linear":
        df, truth = simulate(a.n, a.n_snps, a.theta, a.h2_x, a.gamma_x, a.gamma_y, a.seed)
    elif a.shape == "cox":
        df, truth = simulate_survival(a.n, a.n_snps, a.theta, a.h2_x, a.gamma_x, a.gamma_y, a.seed,
                                      censor_frac=a.censor_frac, followup=a.followup)
    else:
        df, truth = simulate_nonlinear(a.n, a.n_snps, a.shape, a.theta, a.theta2,
                                       a.h2_x, a.gamma_x, a.gamma_y, a.seed)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(a.out.with_name(a.out.name + ".csv"))
    a.out.with_name(a.out.name + ".truth.json").write_text(json.dumps(truth, indent=1))
    print(f"wrote {a.out}.csv ({df.height} rows, {df.width} cols) and {a.out}.truth.json")


if __name__ == "__main__":
    main()
