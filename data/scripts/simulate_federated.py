"""Federated multi-site MR simulator, generalized over the outcome simulator.

Generalizes simulate_federated_sites.py (which is hardwired to a continuous
quadratic outcome) to drive any simulator registered in simulators.py -
continuous or binary, linear/quadratic/threshold - through one shared
interface. Per-site nuisance parameters (population size, h2_x, gamma_x,
gamma_y) are drawn with sample_site_params() from simulate_federated_sites.py
(imported, not duplicated); the causal curve (shape, theta1, theta2) and any
outcome-specific knobs (link, prevalence) are fixed across all sites - same
rationale as simulate_federated_sites.py: the causal effect is biology, not
geography.

simulate_federated_sites.py is left as-is; this is an additive alternative.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from simulate_federated_sites import Heterogeneity, add_heterogeneity_args, sample_site_params  # noqa: E402
from simulators import make_simulator  # noqa: E402


def run_federated(sim_fn, n_sites, n_snps, pop_min, pop_max,
                   h2x_mean, h2x_kappa, gamma_mean, gamma_kappa, seed, out_dir, het=None):
    """het: a Heterogeneity (shared SNPs, site-specific theta1, pleiotropy) or None for the
    default independent-SNP draw. Returns (manifest rows, heterogeneity metadata)."""
    rng = np.random.default_rng(seed)
    n, h2_x, gamma_x, gamma_y = sample_site_params(
        rng, n_sites, pop_min, pop_max, h2x_mean, h2x_kappa, gamma_mean, gamma_kappa
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for i in range(n_sites):
        site_seed = seed + i + 1
        extra = het.extra(site_seed) if het else {}
        fn = sim_fn[i] if isinstance(sim_fn, (list, tuple)) else sim_fn   # one closure per site when theta1 varies
        df, truth = fn(int(n[i]), n_snps, float(h2_x[i]), float(gamma_x[i]), float(gamma_y[i]), site_seed,
                       **extra)
        h2_x[i] = truth.get("h2_x", h2_x[i])          # realized value when SNPs are shared
        site_id = f"site{i + 1:02d}"
        df.write_csv(out_dir / f"{site_id}.csv")
        (out_dir / f"{site_id}.truth.json").write_text(json.dumps(truth, indent=1))

        entry = {
            "site_id": site_id, "n": int(n[i]), "h2_x": float(h2_x[i]),
            "gamma_x": float(gamma_x[i]), "gamma_y": float(gamma_y[i]), "seed": site_seed,
            **({"theta1": float(het.theta1[i])} if het else {}),
        }
        for key in ("avg_slope", "prevalence_realized", "event_rate"):
            if key in truth:
                entry[key] = truth[key]
        manifest.append(entry)

        if "prevalence_realized" in truth:
            extra = f"  prevalence={truth['prevalence_realized']:.3f}"
        elif "event_rate" in truth:
            extra = f"  event_rate={truth['event_rate']:.3f}"
        else:
            extra = ""
        print(f"{site_id}: n={int(n[i]):>6,}  h2_x={h2_x[i]:.3f}  "
              f"gamma_x={gamma_x[i]:.3f}  gamma_y={gamma_y[i]:.3f}{extra}")

    pl.DataFrame(manifest).write_csv(out_dir / "manifest.csv")
    return manifest


def default_out_dir(outcome, shape, link):
    """Encode outcome type (and shape/link where they apply) in the directory name,
    so the provenance of a federated draw is clear from its path alone."""
    if outcome == "continuous":
        tag = f"continuous_{shape}"
    elif outcome == "binary":
        tag = f"binary_{shape}_{link}"
    else:
        tag = "survival"
    return Path(f"simulated_data/federated/{tag}")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--outcome", choices=["continuous", "binary", "survival"], default="continuous")
    p.add_argument("--shape", choices=["linear", "quadratic", "threshold"], default="quadratic",
                   help="form of the X -> outcome link, shared across sites (continuous/binary only)")
    p.add_argument("--theta1", type=float, default=0.3,
                   help="slope/theta1, or the Cox log hazard ratio when --outcome survival")
    p.add_argument("--theta2", type=float, default=0.15, help="quadratic/threshold theta2 (continuous/binary only)")
    p.add_argument("--link", choices=["liability", "logistic"], default="logistic",
                   help="binary outcome only: how liability is turned into 0/1")
    p.add_argument("--prevalence", type=float, default=0.3, help="binary outcome only: target P(Y=1)")
    p.add_argument("--weibull-k", type=float, default=1.5, help="survival only: Weibull baseline hazard shape")
    p.add_argument("--weibull-scale", type=float, default=10.0, help="survival only: Weibull baseline hazard scale")
    p.add_argument("--censor-frac", type=float, default=0.3, help="survival only: target fraction randomly censored")
    p.add_argument("--followup", type=float, default=15.0, help="survival only: administrative end of follow-up")
    p.add_argument("--n-sites", type=int, default=10)
    p.add_argument("--n-snps", type=int, default=20)
    p.add_argument("--pop-min", type=int, default=1_000)
    p.add_argument("--pop-max", type=int, default=10_000)
    p.add_argument("--h2x-mean", type=float, default=0.10)
    p.add_argument("--h2x-kappa", type=float, default=40.0)
    p.add_argument("--gamma-mean", type=float, default=0.3)
    p.add_argument("--gamma-kappa", type=float, default=20.0)
    add_heterogeneity_args(p)
    p.add_argument("--seed", type=int, default=1, help="base seed; site i uses seed + i for its own SNPs/individuals")
    p.add_argument("--out", type=Path, default=None,
                   help="default: simulated_data/federated/<outcome>_<shape>[_<link>]")
    a = p.parse_args()

    out = a.out or default_out_dir(a.outcome, a.shape, a.link)
    het = Heterogeneity(np.random.default_rng(a.seed), a)
    make = lambda t1: make_simulator(a.outcome, a.shape, float(t1), a.theta2, a.link, a.prevalence,
                                     a.weibull_k, a.weibull_scale, a.censor_frac, a.followup)
    # theta1 is closed over by the simulator, so a site-specific theta1 needs one closure per site
    sim_fn = [make(t1) for t1 in het.theta1] if a.theta_sd > 0 else make(a.theta1)
    manifest = run_federated(sim_fn, a.n_sites, a.n_snps, a.pop_min, a.pop_max,
                              a.h2x_mean, a.h2x_kappa, a.gamma_mean, a.gamma_kappa, a.seed, out, het)

    manifest_meta = {
        "outcome": a.outcome, "shape": a.shape if a.outcome != "survival" else None,
        "theta1": a.theta1, "theta2": a.theta2 if a.outcome != "survival" else None,
        "link": a.link if a.outcome == "binary" else None,
        "prevalence": a.prevalence if a.outcome == "binary" else None,
        "weibull_k": a.weibull_k if a.outcome == "survival" else None,
        "weibull_scale": a.weibull_scale if a.outcome == "survival" else None,
        "censor_frac": a.censor_frac if a.outcome == "survival" else None,
        "followup": a.followup if a.outcome == "survival" else None,
        "n_snps": a.n_snps, "seed": a.seed, **het.manifest(), "sites": manifest,
    }
    (out / "manifest.json").write_text(json.dumps(manifest_meta, indent=1))
    print(f"\nwrote {a.n_sites} sites to {out}/ (site01..site{a.n_sites:02d}.csv + .truth.json) "
          f"and {out}/manifest.{{csv,json}}")


if __name__ == "__main__":
    main()
