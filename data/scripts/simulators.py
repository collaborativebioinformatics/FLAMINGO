"""Registry of pluggable simulators for simulate_federated.py.

The federated orchestrator doesn't know or care whether a site's outcome is
continuous or binary, linear or curved - it just calls a `sim_fn` once per
site:

    sim_fn(n, n_snps, h2_x, gamma_x, gamma_y, seed, **extra) -> (df, truth)

`extra` carries the shared-SNP knobs (maf, beta, alpha) that
simulate_federated.py passes when sites share harmonized variants; the
continuous and binary simulators accept them, the survival one does not.

make_simulator() closes over the causal-model knobs that are fixed across
every site (shape, theta1, theta2, and for binary outcomes the link and
target prevalence) and returns exactly that closure. Adding a new outcome
type later means adding one branch here, not touching simulate_federated.py.
"""

import sys
from collections.abc import Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from simulate_basic import simulate, simulate_nonlinear, simulate_survival  # noqa: E402
from simulate_binary import simulate_binary  # noqa: E402


def _continuous_simulator(shape, theta1, theta2) -> Callable[..., tuple]:
    if shape == "linear":
        def sim_fn(n, n_snps, h2_x, gamma_x, gamma_y, seed, **extra) -> tuple:
            df, truth = simulate(n, n_snps, theta1, h2_x, gamma_x, gamma_y, seed, **extra)
            truth.setdefault("avg_slope", theta1)   # the analysis scripts read it for every outcome
            return df, truth
    else:
        def sim_fn(n, n_snps, h2_x, gamma_x, gamma_y, seed, **extra) -> tuple:
            return simulate_nonlinear(n, n_snps, shape, theta1, theta2, h2_x, gamma_x, gamma_y, seed, **extra)
    return sim_fn


def _binary_simulator(shape, theta1, theta2, link, prevalence) -> Callable[..., tuple]:
    def sim_fn(n, n_snps, h2_x, gamma_x, gamma_y, seed, **extra) -> tuple:
        return simulate_binary(n, n_snps, shape, theta1, theta2, h2_x, gamma_x, gamma_y, seed, link, prevalence,
                               **extra)
    return sim_fn


def _survival_simulator(theta, weibull_k, weibull_scale, censor_frac, followup) -> Callable[..., tuple]:
    def sim_fn(n, n_snps, h2_x, gamma_x, gamma_y, seed, **extra) -> tuple:
        if extra.get("alpha") is not None:
            raise ValueError("pleiotropy is not implemented for the survival simulator")
        df, truth = simulate_survival(n, n_snps, theta, h2_x, gamma_x, gamma_y, seed,
                                      weibull_k=weibull_k, weibull_scale=weibull_scale,
                                      censor_frac=censor_frac, followup=followup, **extra)
        truth.setdefault("avg_slope", theta)   # log hazard ratio per unit X, what IVW targets
        return df, truth
    return sim_fn


def make_simulator(outcome, shape="quadratic", theta1=0.3, theta2=0.15, link="logistic", prevalence=0.3,
                    weibull_k=1.5, weibull_scale=10.0, censor_frac=0.3,
                    followup=15.0) -> Callable[..., tuple]:
    """outcome: 'continuous', 'binary', or 'survival'.

    shape ('linear', 'quadratic', or 'threshold') applies to 'continuous' and
    'binary' only - 'survival' is always a Cox proportional-hazards link, with
    theta1 doubling as the log hazard ratio.

    Returns sim_fn(n, n_snps, h2_x, gamma_x, gamma_y, seed) -> (df, truth).
    """
    if outcome == "continuous":
        return _continuous_simulator(shape, theta1, theta2)
    if outcome == "binary":
        return _binary_simulator(shape, theta1, theta2, link, prevalence)
    if outcome == "survival":
        return _survival_simulator(theta1, weibull_k, weibull_scale, censor_frac, followup)
    raise ValueError(f"unknown outcome {outcome!r}; use 'continuous', 'binary', or 'survival'")
