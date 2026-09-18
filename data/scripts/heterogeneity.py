"""Between-site heterogeneity for the federated generators: shared SNP panels,
allele-frequency shifts, site-specific causal effects and pleiotropy.

Owned here so that simulate_federated_sites.py, simulate_federated.py and
fed2sls_sweep.py share one definition. Shared SNPs are the default; every
other knob is off by default.

    --shared-snps       one MAF vector and one beta vector drawn from the base seed and
                        reused at every site; each site's h2_x then follows from those
                        effects at its own allele frequencies (realized, not drawn).
                        Default on; --no-shared-snps gives every site its own variants
                        (how the checked-in sets other than linear_shared were made)
    --maf-shift SD      with --shared-snps, perturb each site's allele frequencies on the
                        logit scale by N(0, SD)
    --theta-sd SD       site-specific causal effect theta1 + N(0, SD)
    --pleiotropy-mean / --pleiotropy-sd
                        direct SNP -> outcome effects alpha_j ~ N(mean, sd) per allele
                        (balanced: mean 0; directional: mean != 0); shared across sites
                        when --shared-snps, drawn per site otherwise
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from simulate_basic import scaled_beta  # noqa: E402


def add_heterogeneity_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--shared-snps", action=argparse.BooleanOptionalAction, default=True,
                   help="same SNPs (MAF, beta) at every site (default); --no-shared-snps draws site-specific variants")
    p.add_argument("--maf-shift", type=float, default=0.0, help="with --shared-snps: logit-scale SD of per-site MAF shifts")
    p.add_argument("--theta-sd", type=float, default=0.0, help="SD of site-specific theta1 around --theta1")
    p.add_argument("--pleiotropy-mean", type=float, default=0.0, help="mean direct SNP -> Y effect per allele")
    p.add_argument("--pleiotropy-sd", type=float, default=0.0, help="SD of direct SNP -> Y effects")


def shift_maf(rng: np.random.Generator, maf: np.ndarray, sd: float) -> np.ndarray:
    """Perturb allele frequencies on the logit scale, kept inside (0.02, 0.98)."""
    if sd <= 0:
        return maf
    logit = np.log(maf / (1 - maf)) + rng.normal(0.0, sd, len(maf))
    return np.clip(1 / (1 + np.exp(-logit)), 0.02, 0.98)


class Heterogeneity:
    """Draws the shared SNP panel, site-specific causal effects and pleiotropy from the
    parsed options `a` (n_sites, n_snps, h2x_mean, theta1, and the knobs above) and hands
    each site its `extra` simulator kwargs."""

    def __init__(self, rng: np.random.Generator, a) -> None:
        self.a = a
        self.pleiotropic = a.pleiotropy_mean != 0 or a.pleiotropy_sd != 0
        self.maf0 = self.beta0 = self.alpha0 = None
        if a.shared_snps:
            self.maf0 = rng.uniform(0.05, 0.5, a.n_snps)
            self.beta0 = scaled_beta(rng, self.maf0, a.h2x_mean)
            self.alpha0 = rng.normal(a.pleiotropy_mean, a.pleiotropy_sd, a.n_snps) if self.pleiotropic else None
        self.theta1 = (a.theta1 + rng.normal(0.0, a.theta_sd, a.n_sites) if a.theta_sd > 0
                       else np.full(a.n_sites, a.theta1))

    def extra(self, site_seed: int) -> dict:
        """Simulator kwargs for one site: shared maf (shifted per site), shared beta, alpha."""
        a = self.a
        site_rng = np.random.default_rng(10_000 + site_seed)
        if a.shared_snps:
            return {"maf": shift_maf(site_rng, self.maf0, a.maf_shift), "beta": self.beta0, "alpha": self.alpha0}
        if self.pleiotropic:
            return {"alpha": site_rng.normal(a.pleiotropy_mean, a.pleiotropy_sd, a.n_snps)}
        return {}

    def manifest(self) -> dict:
        a = self.a
        out = {"shared_snps": a.shared_snps, "maf_shift": a.maf_shift, "theta_sd": a.theta_sd,
               "pleiotropy_mean": a.pleiotropy_mean, "pleiotropy_sd": a.pleiotropy_sd}
        if a.shared_snps:
            out.update({"shared_maf": self.maf0.tolist(), "shared_beta": self.beta0.tolist(),
                        "effect_allele": "allele coded 1 in the dosage; the same allele at every site"})
            if self.alpha0 is not None:
                out["shared_alpha"] = self.alpha0.tolist()
        return out
