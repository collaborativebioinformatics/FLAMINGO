"""Ten federated MR sites sharing one causal X -> Y model.

Reuses the generators in simulate_basic.py (linear, quadratic, threshold or
Cox survival, chosen with --shape) so every site follows the same
G -> X -> Y model (see docs/mr-simulation-model.md); only the per-site
nuisance parameters differ, mimicking ten biobanks in different countries:

    n     population size, spread across [--pop-min, --pop-max]
    h2_x  SNP heritability of X,     Beta-distributed around --h2x-mean
    gamma_x, gamma_y  confounder effects on X and Y, each Beta-distributed
                      around --gamma-mean, drawn independently per site

theta1, theta2 (the causal curve) are fixed across all sites: the causal
effect of X on Y is assumed to be biology, not geography.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from simulate_basic import simulate, simulate_nonlinear, simulate_survival  # noqa: E402


def sample_population_sizes(rng, n_sites, pop_min, pop_max):
    """One draw per equal-width bin spanning [pop_min, pop_max], so ten sites
    cover the whole range instead of clustering around the mean."""
    edges = np.linspace(pop_min, pop_max, n_sites + 1)
    return np.array([rng.integers(lo, hi + 1) for lo, hi in zip(edges[:-1], edges[1:])])


def sample_beta(rng, n_sites, mean, kappa):
    """Beta(mean * kappa, (1 - mean) * kappa): concentration `kappa` controls
    spread around `mean` while keeping draws in (0, 1)."""
    a, b = mean * kappa, (1 - mean) * kappa
    return rng.beta(a, b, size=n_sites)


def sample_site_params(rng, n_sites, pop_min, pop_max, h2x_mean, h2x_kappa, gamma_mean, gamma_kappa):
    n = sample_population_sizes(rng, n_sites, pop_min, pop_max)
    h2_x = sample_beta(rng, n_sites, h2x_mean, h2x_kappa)
    gamma_x = sample_beta(rng, n_sites, gamma_mean, gamma_kappa)
    gamma_y = sample_beta(rng, n_sites, gamma_mean, gamma_kappa)
    # keep Var(e_x) = 1 - h2_x - gamma_x^2 non-negative with margin to spare
    max_gamma_x = np.sqrt(np.maximum(0.95 - h2_x, 0.0))
    gamma_x = np.minimum(gamma_x, max_gamma_x)
    return n, h2_x, gamma_x, gamma_y


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--n-sites", type=int, default=10)
    p.add_argument("--n-snps", type=int, default=20)
    p.add_argument("--pop-min", type=int, default=1_000)
    p.add_argument("--pop-max", type=int, default=10_000)
    p.add_argument("--shape", choices=["linear", "quadratic", "threshold", "cox"], default="quadratic",
                   help="X -> Y link shared by every site")
    p.add_argument("--theta1", type=float, default=0.3,
                   help="slope (linear/cox log HR, quadratic slope at X=0, threshold slope below cutoff)")
    p.add_argument("--theta2", type=float, default=0.15,
                   help="quadratic curvature or threshold cutoff; ignored for linear and cox")
    p.add_argument("--h2x-mean", type=float, default=0.10, help="mean SNP heritability of X across sites")
    p.add_argument("--h2x-kappa", type=float, default=40.0, help="Beta concentration for h2_x (higher = tighter)")
    p.add_argument("--gamma-mean", type=float, default=0.3, help="mean confounder effect (gamma_x, gamma_y)")
    p.add_argument("--gamma-kappa", type=float, default=20.0, help="Beta concentration for gamma_x, gamma_y")
    p.add_argument("--censor-frac", type=float, default=0.3, help="cox: target fraction randomly censored")
    p.add_argument("--followup", type=float, default=15.0, help="cox: administrative end of follow-up")
    p.add_argument("--seed", type=int, default=1, help="base seed; site i uses seed + i for its own SNPs/individuals")
    p.add_argument("--out", type=Path, default=None,
                   help="output directory; default simulated_data/federated/<shape>")
    a = p.parse_args()
    if a.out is None:
        a.out = Path("simulated_data/federated") / a.shape

    rng = np.random.default_rng(a.seed)
    n, h2_x, gamma_x, gamma_y = sample_site_params(
        rng, a.n_sites, a.pop_min, a.pop_max, a.h2x_mean, a.h2x_kappa, a.gamma_mean, a.gamma_kappa
    )

    a.out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for i in range(a.n_sites):
        site_seed = a.seed + i + 1
        common = (int(n[i]), a.n_snps)
        nuisance = (float(h2_x[i]), float(gamma_x[i]), float(gamma_y[i]), site_seed)
        if a.shape == "linear":
            df, truth = simulate(*common, a.theta1, *nuisance)
            truth["avg_slope"] = a.theta1
        elif a.shape == "cox":
            df, truth = simulate_survival(*common, a.theta1, *nuisance,
                                          censor_frac=a.censor_frac, followup=a.followup)
            truth["avg_slope"] = a.theta1
        else:
            df, truth = simulate_nonlinear(*common, a.shape, a.theta1, a.theta2, *nuisance)
        site_id = f"site{i + 1:02d}"
        df.write_csv(a.out / f"{site_id}.csv")
        (a.out / f"{site_id}.truth.json").write_text(json.dumps(truth, indent=1))
        manifest.append({
            "site_id": site_id, "n": int(n[i]), "h2_x": float(h2_x[i]),
            **({"events": int(df["event"].sum())} if a.shape == "cox" else {}),
            "gamma_x": float(gamma_x[i]), "gamma_y": float(gamma_y[i]),
            "avg_slope": truth["avg_slope"], "seed": site_seed,
        })
        print(f"{site_id}: n={int(n[i]):>6,}  h2_x={h2_x[i]:.3f}  "
              f"gamma_x={gamma_x[i]:.3f}  gamma_y={gamma_y[i]:.3f}")

    manifest_df = pl.DataFrame(manifest)
    manifest_df.write_csv(a.out / "manifest.csv")
    (a.out / "manifest.json").write_text(json.dumps({
        "shape": a.shape, "theta1": a.theta1, "theta2": a.theta2, "n_snps": a.n_snps, "seed": a.seed,
        **({"censor_frac": a.censor_frac, "followup": a.followup} if a.shape == "cox" else {}),
        "sites": manifest,
    }, indent=1))
    print(f"\nwrote {a.n_sites} sites to {a.out}/ (site01..site{a.n_sites:02d}.csv + .truth.json) "
          f"and {a.out}/manifest.{{csv,json}}")


if __name__ == "__main__":
    main()
