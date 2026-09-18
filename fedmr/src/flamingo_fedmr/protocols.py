"""The two FedMR protocols, and the quadratic-basis and cross-fitted variants of each.

Both protocols centre every column within the logical site, which absorbs
the site intercept without transmitting a dummy column, and both end in
`statistics.aggregate` + `estimator.fit`. They differ in what instruments
the second stage sees and therefore in how many rounds they need.

SharedInstrument   the same harmonized SNPs at every site (aligned dosages
                   for one effect allele). Z = [G, C], W = [X, C].
                   rounds: 1 (stats)  [+1 robust]
                   Cells of A = sum G_k'G_k receive contributions from every
                   site; that is the layout in the original design note.

LocalFirstStage    each site's SNPs are its own variants (this repo's default
                   simulator), so there is no shared first stage to learn.
                   The site fits X ~ [1, G, C] locally, forms xhat, and the
                   second stage uses Z = [xhat, C], W = [X, C]: a generated
                   instrument whose cross-products every site contributes to
                   the same small ((1+q) x (1+q)) matrix. The result equals
                   pooled 2SLS with per-site first stages and site intercepts,
                   i.e. `pooled_2sls` in the data scripts. (Giving each site's
                   xhat its own column instead would be a different, larger
                   instrument set: identical for one endogenous regressor,
                   not for the quadratic basis.)
                   rounds: 1 (stats)  [+1 robust]

Quadratic basis    W = [X, X^2, C], Z = [xhat, xhat^2, C]: the generated-
                   instrument form used by `quadratic_2sls` in the data
                   scripts. Under LocalFirstStage the first stage is local so
                   it stays at 1 round. Under SharedInstrument the first
                   stage is global: round 1 sums (Z1'Z1, Z1'X), the coordinator
                   broadcasts pi, round 2 sums the second-stage statistics.
                   rounds: 1 or 2  [+1 robust]

Cross-fitting      out-of-fold generated instrument. The estimating equation is

                       sum_i  xhat_i^(-fold(i)) (Y_i - theta X_i) = 0

                   (with C partialled the same way): xhat_oof is the
                   *instrument* for X, not a substituted regressor. Under
                   LocalFirstStage the folds are local, 1 round. Under
                   SharedInstrument, round 1 sums per-fold first-stage moments,
                   the coordinator returns pi_j from (A - A_j, b - b_j), round 2
                   sums the second-stage statistics.  rounds: 1 or 2  [+1 robust]
                   Whether this reduces the one-sample weak-instrument lean is
                   a hypothesis the sweep tests, not a property assumed here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import SiteData
from .estimator import FedMRResult, fit, robust_cov
from .schema import Design, Role, centre_within
from .statistics import Stats, aggregate, aggregate_robust, site_robust_stats, site_stats


# ----------------------------------------------------------------------------- local pieces


def _fs_matrix(site: SiteData) -> np.ndarray:
    """[G, C] without a constant: the site intercept is absorbed by centring, or fitted locally."""
    return np.column_stack([site.G, site.C])


def local_first_stage(site: SiteData) -> np.ndarray:
    """OLS of X on [1, G, C] at the site; returns xhat."""
    Z1 = np.column_stack([np.ones(site.n), _fs_matrix(site)])
    return Z1 @ np.linalg.lstsq(Z1, site.X, rcond=None)[0]


def local_first_stage_diagnostics(
    site: SiteData,
    xhat: np.ndarray | None = None,
) -> dict[str, float | int]:
    """Residual sums of squares of X ~ [1, G, C] and X ~ [1, C] at the site, so the coordinator
    can report the F of the original SNP set (these sums are additive across sites)."""
    full = np.column_stack([np.ones(site.n), _fs_matrix(site)])
    if xhat is None:
        xhat = full @ np.linalg.lstsq(full, site.X, rcond=None)[0]
    reduced = np.column_stack([np.ones(site.n), site.C])
    reduced_xhat = reduced @ np.linalg.lstsq(reduced, site.X, rcond=None)[0]
    return {
        "rss_full": float(np.sum((site.X - xhat) ** 2)),
        "rss_reduced": float(np.sum((site.X - reduced_xhat) ** 2)),
        "n_instruments": site.G.shape[1],
        "n_params": full.shape[1],
    }


def _cov_cols(site: SiteData) -> tuple[list[str], list[Role]]:
    return [f"cov:{c}" for c in site.cov_names], [Role.EXOGENOUS] * len(site.cov_names)


def design_shared(site: SiteData) -> Design:
    """Z = [G, C], W = [X, C], all centred within site."""
    G, X, Y, C = centre_within(site.G, site.X, site.Y, site.C)
    cn, cr = _cov_cols(site)
    m = site.G.shape[1]
    return Design(site.name, np.column_stack([G, C]), np.column_stack([X, C]), Y,
                  [f"snp{j}" for j in range(m)] + cn, ["X"] + cn,
                  [Role.INSTRUMENT] * m + cr, [Role.ENDOGENOUS] + cr)


def design_generated(
    site: SiteData,
    xhat: np.ndarray,
    basis: str = "linear",
    first_stage_local: dict[str, float | int] | None = None,
) -> Design:
    """Second stage on a generated instrument, common columns at every site.
    basis 'linear': Z = [xhat, C], W = [X, C]; 'quadratic': Z = [xhat, xhat^2, C],
    W = [X, X^2, C]. Centred within site. first_stage_local carries the site's own
    first-stage diagnostics when it fitted one (see local_first_stage_diagnostics)."""
    cn, cr = _cov_cols(site)
    if basis == "linear":
        xh, X, Y, C = centre_within(xhat, site.X, site.Y, site.C)
        return Design(site.name, np.column_stack([xh, C]), np.column_stack([X, C]), Y,
                      ["xhat"] + cn, ["X"] + cn, [Role.INSTRUMENT] + cr, [Role.ENDOGENOUS] + cr,
                      first_stage_local=first_stage_local)
    if basis == "quadratic":
        xh, xh2, X, X2, Y, C = centre_within(xhat, xhat**2, site.X, site.X**2, site.Y, site.C)
        return Design(site.name, np.column_stack([xh, xh2, C]), np.column_stack([X, X2, C]), Y,
                      ["xhat", "xhat2"] + cn, ["X", "X2"] + cn,
                      [Role.INSTRUMENT] * 2 + cr, [Role.ENDOGENOUS] * 2 + cr, first_stage_local=first_stage_local)
    raise ValueError(basis)


# ----------------------------------------------------------------------------- shared first stage (SharedInstrument + quadratic / cross-fit)


def first_stage_moments(site: SiteData) -> tuple[np.ndarray, np.ndarray, int]:
    """(Z1'Z1, Z1'X, n) for a global first stage X ~ [G, C] + site intercepts, the intercepts
    absorbed by centring within site. Shared SNPs only."""
    Z1, X = centre_within(_fs_matrix(site), site.X)
    return Z1.T @ Z1, Z1.T @ X, site.n


def first_stage_global(moments: list[tuple[np.ndarray, np.ndarray, int]]) -> np.ndarray:
    A1 = sum(m[0] for m in moments)
    b1 = sum(m[1] for m in moments)
    return np.linalg.solve(A1, b1)


def predict_xhat(site: SiteData, pi: np.ndarray) -> np.ndarray:
    """xhat = a_k + [G, C] pi with the site intercept a_k = mean(X) - mean([G, C]) pi recovered
    locally. The intercept matters once xhat is squared: (a_k + z pi)^2 has a 2 a_k z pi term
    that within-site centring does not absorb, so dropping a_k would change the instrument set."""
    Z1 = _fs_matrix(site)
    return site.X.mean() + (Z1 - Z1.mean(axis=0)) @ pi


def fold_ids(n: int, k: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).permutation(n) % k


def crossfit_moments(site: SiteData, folds: np.ndarray, k: int) -> list[tuple[np.ndarray, np.ndarray, int]]:
    """Per-fold first-stage moments, centred within (site, fold) so the fixed effect of the
    held-out fold never enters the other folds' fit: nothing from fold j leaks into pi_j."""
    Z1 = _fs_matrix(site)
    out = []
    for j in range(k):
        Zj, Xj = centre_within(Z1[folds == j], site.X[folds == j])
        out.append((Zj.T @ Zj, Zj.T @ Xj, int((folds == j).sum())))
    return out


def crossfit_pi(
    per_site_fold_moments: list[list[tuple[np.ndarray, np.ndarray, int]]],
    k: int,
) -> list[np.ndarray]:
    """pi_j from the moments of every fold except j, over every site: (A - A_j, b - b_j)."""
    A = sum(m[0] for ms in per_site_fold_moments for m in ms)
    b = sum(m[1] for ms in per_site_fold_moments for m in ms)
    out = []
    for j in range(k):
        Aj = sum(ms[j][0] for ms in per_site_fold_moments)
        bj = sum(ms[j][1] for ms in per_site_fold_moments)
        out.append(np.linalg.solve(A - Aj, b - bj))
    return out


def crossfit_xhat(site: SiteData, folds: np.ndarray, pis: list[np.ndarray]) -> np.ndarray:
    """Out-of-fold xhat_j = a_j + [G, C] pi_j, with the site intercept a_j estimated on the
    site's *other* folds (mean of X - [G, C] pi_j there), so fold j contributes nothing to
    its own instrument beyond its genotypes."""
    Z1 = _fs_matrix(site)
    xhat = np.empty(site.n)
    for j, pi in enumerate(pis):
        other = folds != j
        a_j = float(np.mean(site.X[other] - Z1[other] @ pi))
        xhat[folds == j] = a_j + Z1[folds == j] @ pi
    return xhat


def crossfit_xhat_local(site: SiteData, folds: np.ndarray, k: int) -> np.ndarray:
    """Site-local cross-fitting: X ~ [1, G, C] on the other folds, predicted on fold j."""
    Z1 = np.column_stack([np.ones(site.n), _fs_matrix(site)])
    xhat = np.empty(site.n)
    for j in range(k):
        tr = folds != j
        xhat[folds == j] = Z1[folds == j] @ np.linalg.lstsq(Z1[tr], site.X[tr], rcond=None)[0]
    return xhat


# ----------------------------------------------------------------------------- protocols


@dataclass
class Run:
    """A completed protocol run: the designs (never leave the sites), the sums, and the result."""
    designs: list[Design]
    stats: Stats
    result: FedMRResult
    rounds: int


def _finish(designs: list[Design], robust: bool, extra_rounds: int = 0) -> Run:
    """Round 1 (statistics) and, if asked, round 2 (robust covariance) over the given designs."""
    stats = aggregate([site_stats(d) for d in designs])
    res = fit(stats)
    rounds = 1 + extra_rounds
    if robust:
        tb = res.theta_by_name()
        H = aggregate_robust([(d.z_names, site_robust_stats(d, tb)) for d in designs], stats.layout)
        res.robust_cov = robust_cov(stats, H)
        rounds += 1
    return Run(designs, stats, res, rounds)


class SharedInstrumentFedMR:
    """Harmonized SNPs at every site. basis 'linear' uses G directly (one round);
    'quadratic' or crossfit > 0 need a global first stage (two rounds)."""

    def __init__(self, basis: str = "linear", crossfit: int = 0, robust: bool = True, seed: int = 0) -> None:
        self.basis, self.crossfit, self.robust, self.seed = basis, crossfit, robust, seed

    def run(self, sites: list[SiteData]) -> Run:
        if self.crossfit:
            folds = [fold_ids(s.n, self.crossfit, self.seed + i) for i, s in enumerate(sites)]
            pis = crossfit_pi([crossfit_moments(s, f, self.crossfit) for s, f in zip(sites, folds)], self.crossfit)
            designs = [design_generated(s, crossfit_xhat(s, f, pis), self.basis) for s, f in zip(sites, folds)]
            return _finish(designs, self.robust, extra_rounds=1)
        if self.basis == "quadratic":
            pi = first_stage_global([first_stage_moments(s) for s in sites])
            designs = [design_generated(s, predict_xhat(s, pi), "quadratic") for s in sites]
            return _finish(designs, self.robust, extra_rounds=1)
        return _finish([design_shared(s) for s in sites], self.robust)


class LocalFirstStageFedMR:
    """Site-specific SNPs: local first stage, generated instrument, one round (+ robust).
    Each site also releases the residual sums of squares of its first stage so the
    coordinator reports the F of the SNP set, not of the single generated column."""

    def __init__(self, basis: str = "linear", crossfit: int = 0, robust: bool = True, seed: int = 0) -> None:
        self.basis, self.crossfit, self.robust, self.seed = basis, crossfit, robust, seed

    def run(self, sites: list[SiteData]) -> Run:
        if self.crossfit:
            xhats = [crossfit_xhat_local(s, fold_ids(s.n, self.crossfit, self.seed + i), self.crossfit)
                     for i, s in enumerate(sites)]
            diagnostics = [local_first_stage_diagnostics(site) for site in sites]
        else:
            xhats = [local_first_stage(site) for site in sites]
            diagnostics = [local_first_stage_diagnostics(site, xhat) for site, xhat in zip(sites, xhats)]
        designs = [design_generated(site, xhat, self.basis, first_stage)
                   for site, xhat, first_stage in zip(sites, xhats, diagnostics)]
        return _finish(designs, self.robust)


def protocol_for(manifest: dict, **kw) -> SharedInstrumentFedMR | LocalFirstStageFedMR:
    """Pick the protocol from a dataset manifest: shared_snps -> SharedInstrument, else LocalFirstStage."""
    return SharedInstrumentFedMR(**kw) if manifest.get("shared_snps") else LocalFirstStageFedMR(**kw)


def fedmr(sites: list[SiteData], instruments: str = "site", **kw) -> FedMRResult:
    """One-call convenience: instruments 'site' (LocalFirstStage) or 'shared' (SharedInstrument)."""
    proto = SharedInstrumentFedMR(**kw) if instruments == "shared" else LocalFirstStageFedMR(**kw)
    return proto.run(sites).result
