"""FedMR: exact federated one-sample Mendelian randomization from sufficient statistics.

Pooled two-stage least squares (2SLS) with structural regressors W = [X, C]
and instruments Z = [G, C] is

    theta = (B' A^-1 B)^-1 B' A^-1 c,     A = Z'Z,  B = Z'W,  c = Z'Y

and every one of those matrices is a sum over rows, so a sum over sites.
Each site computes its own A_k, B_k, c_k (plus D_k = W'W, e_k = W'Y,
f_k = Y'Y and n_k for standard errors and diagnostics), the coordinator sums
them, and the result is the pooled estimator to machine precision. Nothing
is trained and no individual row leaves a site.

Column alignment. Every column of Z and W carries a name. Columns with the
same name at different sites are summed into one global column; a column
whose name is unique to a site (for instance "site03:snp0" when each site
has its own SNPs) gets its own global column, which makes the global A
block-diagonal in those columns. That one rule covers both layouts:

    site-specific instruments   Z_k = [site_k:const, site_k:snp0..]   (block-diagonal A)
    shared, harmonized SNPs     Z_k = [snp0.., site dummies]           (dense A)

Site dummies appear in both W and Z, so the causal effect is estimated
within site, as the design note recommends.

Protocol (what leaves a site):
    round 1   SiteStats: A, B, c, D, e, f, n and the column names
              -> theta, classical covariance, first-stage F and partial R^2
    round 2   H_k = Z' diag(u^2) Z with u = Y - W theta       (optional)
              -> heteroskedasticity-robust covariance
    crossfit  per-fold first-stage moments, then out-of-fold X_hat   (optional, shared SNPs)

The functions are pure numpy so the same code runs inside the data scripts,
the in-process engine and the NVFlare client and server.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ORACLE_COLUMNS = ("U",)     # simulator oracles that must never enter a design matrix


# ----------------------------------------------------------------------------- data


@dataclass
class SiteData:
    """One site's individual-level data. G: (n, m) dosages, X: (n,), Y: (n,), C: (n, q) covariates."""
    name: str
    G: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    C: np.ndarray = None
    cov_names: tuple = ()

    def __post_init__(self):
        if self.C is None:
            self.C = np.empty((len(self.X), 0))
        assert self.G.shape[0] == len(self.X) == len(self.Y) == self.C.shape[0]
        for name in self.cov_names:
            if name in ORACLE_COLUMNS:
                raise ValueError(f"{name!r} is a simulator oracle and may not be used as a covariate")

    @property
    def n(self):
        return len(self.X)


def load_site_csv(path, name=None, covariates=()) -> SiteData:
    """Read a simulated site CSV (id, snp*, U, X, Y) with polars or pandas, whichever is installed."""
    path = Path(path)
    try:
        import polars as pl
        df = pl.read_csv(path)
        cols = df.columns
        get = lambda c: df[c].to_numpy()
    except ImportError:
        import pandas as pd
        df = pd.read_csv(path)
        cols = list(df.columns)
        get = lambda c: df[c].to_numpy()
    snp_cols = [c for c in cols if c.startswith("snp")]
    G = np.column_stack([get(c) for c in snp_cols]).astype(np.float64)
    C = np.column_stack([get(c) for c in covariates]).astype(np.float64) if covariates else None
    return SiteData(name or path.stem, G, get("X").astype(np.float64), get("Y").astype(np.float64), C,
                    tuple(covariates))


def load_sites(folder, covariates=()) -> list[SiteData]:
    return [load_site_csv(p, covariates=covariates) for p in sorted(Path(folder).glob("site*.csv"))]


# ----------------------------------------------------------------------------- design matrices


@dataclass
class Design:
    """Z, W, Y for one site plus the column names used for alignment.
    `excluded` are the Z columns that are instruments proper (not also in W)."""
    Z: np.ndarray
    W: np.ndarray
    Y: np.ndarray
    z_names: list
    w_names: list
    excluded: list
    endogenous: list


def _first_stage_local(G, X):
    Z1 = np.column_stack([np.ones(len(X)), G])
    return Z1 @ np.linalg.lstsq(Z1, X, rcond=None)[0]


def build_design(site: SiteData, instruments="site", basis="linear", xhat=None) -> Design:
    """Design matrices for one site.

    instruments  "site"    the site's SNPs are its own variants: Z gets per-site columns
                           (site:const, site:snp_j); the global A is block-diagonal
                 "shared"  the same harmonized SNPs everywhere: Z gets shared columns snp_j
                           and the site's dummy column
    basis        "linear"     W = [X, site dummy],  Z as above (one round, no first stage)
                 "quadratic"  W = [X, X^2, site dummy], Z = [xhat, xhat^2, site dummy] with
                              xhat the SNP-predicted X: site-local OLS by default, or the
                              `xhat` argument (a global first stage, see first_stage_global)
    Site dummies are always in W (and, for shared/quadratic layouts, in Z) so the
    estimate is within site.
    """
    n, k = site.n, site.name
    dummy = np.ones(n)
    cov_names = [f"cov:{c}" for c in site.cov_names]
    if basis == "linear":
        if instruments == "site":
            Z = np.column_stack([dummy, site.G, site.C])
            z_names = [f"{k}:const", *[f"{k}:snp{j}" for j in range(site.G.shape[1])], *cov_names]
            excluded = z_names[1:1 + site.G.shape[1]]
        elif instruments == "shared":
            Z = np.column_stack([site.G, dummy, site.C])
            z_names = [*[f"snp{j}" for j in range(site.G.shape[1])], f"site:{k}", *cov_names]
            excluded = z_names[:site.G.shape[1]]
        else:
            raise ValueError(instruments)
        W = np.column_stack([site.X, dummy, site.C])
        w_names = ["X", f"site:{k}", *cov_names]
        endogenous = ["X"]
    elif basis == "quadratic":
        if xhat is None:
            xhat = _first_stage_local(site.G, site.X)
        Z = np.column_stack([xhat, xhat**2, dummy, site.C])
        z_names = ["xhat", "xhat2", f"site:{k}", *cov_names]
        W = np.column_stack([site.X, site.X**2, dummy, site.C])
        w_names = ["X", "X2", f"site:{k}", *cov_names]
        excluded, endogenous = ["xhat", "xhat2"], ["X", "X2"]
    else:
        raise ValueError(basis)
    return Design(Z, W, site.Y, z_names, w_names, excluded, endogenous)


# ----------------------------------------------------------------------------- sufficient statistics


@dataclass
class SiteStats:
    """What a site sends in round 1. Symmetric A and D; only the column names identify columns."""
    site: str
    n: int
    z_names: list
    w_names: list
    excluded: list
    endogenous: list
    A: np.ndarray
    B: np.ndarray
    c: np.ndarray
    D: np.ndarray
    e: np.ndarray
    f: float

    def arrays(self):
        """Numeric payload as a dict of arrays, for transport (e.g. as NVFlare FLModel params)."""
        return {"A": self.A, "B": self.B, "c": self.c, "D": self.D, "e": self.e,
                "f": np.array([self.f]), "n": np.array([self.n], dtype=np.float64)}

    def meta(self):
        return {"site": self.site, "z_names": list(self.z_names), "w_names": list(self.w_names),
                "excluded": list(self.excluded), "endogenous": list(self.endogenous)}

    @classmethod
    def from_transport(cls, arrays, meta):
        return cls(meta["site"], int(round(float(arrays["n"][0]))), list(meta["z_names"]), list(meta["w_names"]),
                   list(meta["excluded"]), list(meta["endogenous"]),
                   np.asarray(arrays["A"], float), np.asarray(arrays["B"], float), np.asarray(arrays["c"], float),
                   np.asarray(arrays["D"], float), np.asarray(arrays["e"], float), float(np.asarray(arrays["f"]).ravel()[0]))


def site_stats(site: SiteData, instruments="site", basis="linear", xhat=None) -> SiteStats:
    d = build_design(site, instruments, basis, xhat)
    return stats_from_design(site.name, d)


def stats_from_design(name, d: Design) -> SiteStats:
    Z, W, Y = d.Z, d.W, d.Y
    return SiteStats(name, len(Y), list(d.z_names), list(d.w_names), list(d.excluded), list(d.endogenous),
                     Z.T @ Z, Z.T @ W, Z.T @ Y, W.T @ W, W.T @ Y, float(Y @ Y))


def _union(names_per_site):
    out = []
    for names in names_per_site:
        for nm in names:
            if nm not in out:
                out.append(nm)
    return out


@dataclass
class Stats:
    """The coordinator's sums, with the global column layout."""
    N: int
    z_names: list
    w_names: list
    excluded: list
    endogenous: list
    A: np.ndarray
    B: np.ndarray
    c: np.ndarray
    D: np.ndarray
    e: np.ndarray
    f: float
    sites: list = field(default_factory=list)


def aggregate(parts: list[SiteStats]) -> Stats:
    """Sum the sites' statistics into the global layout (union of column names, in order of first use)."""
    z_names = _union(p.z_names for p in parts)
    w_names = _union(p.w_names for p in parts)
    excluded = _union(p.excluded for p in parts)
    endogenous = _union(p.endogenous for p in parts)
    zi = {nm: i for i, nm in enumerate(z_names)}
    wi = {nm: i for i, nm in enumerate(w_names)}
    A = np.zeros((len(z_names), len(z_names)))
    B = np.zeros((len(z_names), len(w_names)))
    c = np.zeros(len(z_names))
    D = np.zeros((len(w_names), len(w_names)))
    e = np.zeros(len(w_names))
    f, N = 0.0, 0
    for p in parts:
        zr = np.array([zi[nm] for nm in p.z_names])
        wr = np.array([wi[nm] for nm in p.w_names])
        A[np.ix_(zr, zr)] += p.A
        B[np.ix_(zr, wr)] += p.B
        c[zr] += p.c
        D[np.ix_(wr, wr)] += p.D
        e[wr] += p.e
        f += p.f
        N += p.n
    return Stats(N, z_names, w_names, excluded, endogenous, A, B, c, D, e, f, [p.site for p in parts])


# ----------------------------------------------------------------------------- the estimator


def _solve(A, b):
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(A) @ b


@dataclass
class FedMRResult:
    theta: np.ndarray
    cov: np.ndarray               # classical 2SLS covariance
    w_names: list
    N: int
    rss: float
    sigma2: float
    first_stage: dict             # per endogenous column: {"F", "partial_r2", "df1", "df2"}
    robust_cov: np.ndarray = None

    def __getitem__(self, name):
        return float(self.theta[self.w_names.index(name)])

    def se(self, name, robust=False):
        i = self.w_names.index(name)
        cov = self.robust_cov if robust else self.cov
        return float(np.sqrt(cov[i, i]))

    def summary(self, names=None):
        names = names or [n for n in self.w_names if not n.startswith("site:")]
        return {n: (self[n], self.se(n), self.se(n, True) if self.robust_cov is not None else None) for n in names}


def fit(s: Stats) -> FedMRResult:
    """theta = (B'A^-1 B)^-1 B'A^-1 c, classical covariance sigma2 (B'A^-1 B)^-1, first-stage diagnostics."""
    AinvB = _solve(s.A, s.B)                      # A^-1 B
    M = s.B.T @ AinvB                             # B' A^-1 B
    v = AinvB.T @ s.c                             # B' A^-1 c
    theta = _solve(M, v)
    r = len(theta)
    rss = s.f - 2 * theta @ s.e + theta @ s.D @ theta
    sigma2 = rss / (s.N - r)
    Minv = np.linalg.pinv(M) if r > 0 else M
    cov = sigma2 * Minv
    return FedMRResult(theta, cov, list(s.w_names), s.N, float(rss), float(sigma2), first_stage_diagnostics(s))


def first_stage_diagnostics(s: Stats) -> dict:
    """Per endogenous column x: partial F and partial R^2 of the excluded instruments,
    conditional on the exogenous Z columns. Everything comes from A, B, D."""
    zi = {nm: i for i, nm in enumerate(s.z_names)}
    wi = {nm: i for i, nm in enumerate(s.w_names)}
    exog = [i for nm, i in zi.items() if nm not in s.excluded]
    m = len(s.excluded)
    out = {}
    for x in s.endogenous:
        bx = s.B[:, wi[x]]
        xx = s.D[wi[x], wi[x]]
        rss_full = xx - bx @ _solve(s.A, bx)
        if exog:
            Ae, be = s.A[np.ix_(exog, exog)], bx[exog]
            rss_red = xx - be @ _solve(Ae, be)
        else:
            rss_red = xx
        df2 = s.N - len(s.z_names)
        F = ((rss_red - rss_full) / m) / (rss_full / df2)
        out[x] = {"F": float(F), "partial_r2": float((rss_red - rss_full) / rss_red), "df1": m, "df2": int(df2),
                  "pi": _solve(s.A, bx)}
    return out


# ----------------------------------------------------------------------------- round 2: robust covariance


def site_robust_stats(d: Design, theta_by_name: dict) -> np.ndarray:
    """H_k = Z' diag(u^2) Z with u = Y - W theta, theta aligned to this site's W columns."""
    theta = np.array([theta_by_name[nm] for nm in d.w_names])
    u = d.Y - d.W @ theta
    return d.Z.T @ (d.Z * (u**2)[:, None])


def aggregate_robust(parts: list[tuple[list, np.ndarray]], z_names) -> np.ndarray:
    """Sum the sites' H_k into the global Z layout. parts: [(site z_names, H_k), ...]."""
    zi = {nm: i for i, nm in enumerate(z_names)}
    H = np.zeros((len(z_names), len(z_names)))
    for names, Hk in parts:
        zr = np.array([zi[nm] for nm in names])
        H[np.ix_(zr, zr)] += Hk
    return H


def robust_cov(s: Stats, H: np.ndarray) -> np.ndarray:
    """Sandwich: M^-1 B'A^-1 H A^-1 B M^-1."""
    AinvB = _solve(s.A, s.B)
    M = s.B.T @ AinvB
    Minv = np.linalg.pinv(M)
    return Minv @ AinvB.T @ H @ AinvB @ Minv


def theta_by_name(res: FedMRResult) -> dict:
    return dict(zip(res.w_names, map(float, res.theta)))


# ----------------------------------------------------------------------------- shared-SNP first stage


def first_stage_moments(site: SiteData):
    """Round-1 moments for a global first stage X ~ [1, G]: (Z1'Z1, Z1'X, n). Shared-SNP layout only."""
    Z1 = np.column_stack([np.ones(site.n), site.G])
    return Z1.T @ Z1, Z1.T @ site.X, site.n


def first_stage_global(moments):
    """pi from summed moments; predicting xhat = [1, G] pi at every site is then local."""
    A1 = sum(m[0] for m in moments)
    b1 = sum(m[1] for m in moments)
    return _solve(A1, b1)


def predict_xhat(site: SiteData, pi):
    return np.column_stack([np.ones(site.n), site.G]) @ pi


# ----------------------------------------------------------------------------- global standardization


def global_moments(sites: list[SiteData]):
    """Sum G, G^2 and n over sites -> (mean, sd) per SNP, so every site can apply one transform.
    Never standardize per site: that changes the pooled estimand."""
    s1 = sum(site.G.sum(axis=0) for site in sites)
    s2 = sum((site.G**2).sum(axis=0) for site in sites)
    N = sum(site.n for site in sites)
    mean = s1 / N
    sd = np.sqrt(s2 / N - mean**2)
    return mean, sd


# ----------------------------------------------------------------------------- cross-fitting


def fold_ids(n, k, seed=0):
    rng = np.random.default_rng(seed)
    return rng.permutation(n) % k


def crossfit_moments(site: SiteData, folds: np.ndarray, k: int):
    """Per-fold first-stage moments (Z1'Z1, Z1'X, n) restricted to fold j, for j in 0..k-1."""
    Z1 = np.column_stack([np.ones(site.n), site.G])
    return [(Z1[folds == j].T @ Z1[folds == j], Z1[folds == j].T @ site.X[folds == j], int((folds == j).sum()))
            for j in range(k)]


def crossfit_pi(per_site_fold_moments, k):
    """Out-of-fold global first stage: pi_j from (A - A_j, b - b_j). Nothing but moments moves."""
    A = sum(m[0] for ms in per_site_fold_moments for m in ms)
    b = sum(m[1] for ms in per_site_fold_moments for m in ms)
    pis = []
    for j in range(k):
        Aj = sum(ms[j][0] for ms in per_site_fold_moments)
        bj = sum(ms[j][1] for ms in per_site_fold_moments)
        pis.append(_solve(A - Aj, b - bj))
    return pis


def crossfit_xhat(site: SiteData, folds, pis):
    Z1 = np.column_stack([np.ones(site.n), site.G])
    xhat = np.empty(site.n)
    for j, pi in enumerate(pis):
        xhat[folds == j] = Z1[folds == j] @ pi
    return xhat


def crossfit_xhat_local(site: SiteData, folds, k):
    """Site-local cross-fitting: first stage on the other folds of the same site."""
    Z1 = np.column_stack([np.ones(site.n), site.G])
    xhat = np.empty(site.n)
    for j in range(k):
        tr = folds != j
        pi = np.linalg.lstsq(Z1[tr], site.X[tr], rcond=None)[0]
        xhat[folds == j] = Z1[folds == j] @ pi
    return xhat


def build_design_xhat(site: SiteData, xhat, basis="linear") -> Design:
    """Second-stage design with a given X_hat as the instrument (cross-fitted or global first stage):
    Z = [xhat(, xhat^2), site dummy], W = [X(, X^2), site dummy]."""
    if basis == "quadratic":
        return build_design(site, "shared", "quadratic", xhat=xhat)
    n, k = site.n, site.name
    dummy = np.ones(n)
    cov_names = [f"cov:{c}" for c in site.cov_names]
    Z = np.column_stack([xhat, dummy, site.C])
    W = np.column_stack([site.X, dummy, site.C])
    return Design(Z, W, site.Y, ["xhat", f"site:{k}", *cov_names], ["X", f"site:{k}", *cov_names], ["xhat"], ["X"])


# ----------------------------------------------------------------------------- one-call convenience


def fedmr(sites: list[SiteData], instruments="site", basis="linear", robust=True, crossfit=0, seed=0) -> FedMRResult:
    """Run the whole protocol in-process. Sites only ever expose their statistics to `aggregate`.

    crossfit = k > 0 uses out-of-fold X_hat as the instrument: a global first stage over
    the other folds of every site when instruments == "shared", a site-local one otherwise.
    """
    if crossfit:
        folds = [fold_ids(s.n, crossfit, seed + i) for i, s in enumerate(sites)]
        if instruments == "shared":
            pis = crossfit_pi([crossfit_moments(s, f, crossfit) for s, f in zip(sites, folds)], crossfit)
            xhats = [crossfit_xhat(s, f, pis) for s, f in zip(sites, folds)]
        else:
            xhats = [crossfit_xhat_local(s, f, crossfit) for s, f in zip(sites, folds)]
        designs = [build_design_xhat(s, xh, basis) for s, xh in zip(sites, xhats)]
    elif basis == "quadratic" and instruments == "shared":
        pi = first_stage_global([first_stage_moments(s) for s in sites])
        designs = [build_design(s, "shared", "quadratic", xhat=predict_xhat(s, pi)) for s in sites]
    else:
        designs = [build_design(s, instruments, basis) for s in sites]
    stats = aggregate([stats_from_design(s.name, d) for s, d in zip(sites, designs)])
    res = fit(stats)
    if robust:
        tb = theta_by_name(res)
        H = aggregate_robust([(d.z_names, site_robust_stats(d, tb)) for d in designs], stats.z_names)
        res.robust_cov = robust_cov(stats, H)
    return res
