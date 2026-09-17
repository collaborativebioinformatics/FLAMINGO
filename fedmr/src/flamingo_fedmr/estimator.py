"""The 2SLS solve on summed statistics, with covariances and diagnostics.

    theta = (B' A^-1 B)^-1 B' A^-1 c
    RSS   = f - 2 theta'e + theta' D theta
    Var   = RSS / (N - r - absorbed) * (B' A^-1 B)^-1          classical
    Var   = M^-1 B'A^-1 H A^-1 B M^-1,  M = B' A^-1 B          HC0 sandwich (round 2)

No explicit inverses of A: every A^-1 (.) is a linear solve. M is r x r
(r = number of structural regressors), and its inverse is obtained by
solving against the identity because the covariance needs it explicitly.
Rank and conditioning are reported and under-identification or singularity
raise with the reason.

First-stage diagnostics are per endogenous column: the partial F and partial
R^2 of the excluded instruments given the exogenous columns. With one
endogenous regressor this is the usual first-stage F. With several (for
instance X and X^2) it is *not* a conditional (Sanderson-Windmeijer) F and
should not be read as one; the result says so.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .statistics import Stats


class IdentificationError(ValueError):
    pass


@dataclass
class Diagnostics:
    rank_A: int
    dim_A: int
    cond_A: float
    cond_M: float
    n_endogenous: int
    n_instruments: int
    n_exogenous: int
    absorbed: int
    first_stage: dict          # per endogenous column: F, partial_r2, df1, df2, conditional (bool)


@dataclass
class FedMRResult:
    theta: np.ndarray
    cov: np.ndarray
    w_names: list
    N: int
    rss: float
    sigma2: float
    df_resid: int
    diagnostics: Diagnostics
    robust_cov: np.ndarray = None

    def __getitem__(self, name):
        return float(self.theta[self.w_names.index(name)])

    def se(self, name, robust=False):
        i = self.w_names.index(name)
        cov = self.robust_cov if robust else self.cov
        if cov is None:
            raise ValueError("robust covariance not computed; run the robust round")
        return float(np.sqrt(cov[i, i]))

    def theta_by_name(self):
        return dict(zip(self.w_names, map(float, self.theta)))


def _solve(A, b, what):
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError as err:
        raise IdentificationError(f"{what} is singular: {err}") from None


def fit(s: Stats) -> FedMRResult:
    lay = s.layout
    p, m, q = len(lay.endogenous), len(lay.instruments), len(lay.exogenous)
    if m < p:
        raise IdentificationError(f"under-identified: {p} endogenous regressors but {m} excluded instruments")
    dim_A = s.A.shape[0]
    rank_A = int(np.linalg.matrix_rank(s.A))
    if rank_A < dim_A:
        raise IdentificationError(f"Z'Z has rank {rank_A} < {dim_A}: collinear or constant instrument columns")
    AinvB = _solve(s.A, s.B, "Z'Z")
    M = s.B.T @ AinvB
    v = AinvB.T @ s.c
    theta = _solve(M, v, "B'A^-1B")
    r = len(theta)
    rss = float(s.f - 2 * theta @ s.e + theta @ s.D @ theta)
    df_resid = s.N - r - lay.absorbed
    if df_resid <= 0:
        raise IdentificationError(f"no residual degrees of freedom: N={s.N}, r={r}, absorbed={lay.absorbed}")
    sigma2 = rss / df_resid
    Minv = _solve(M, np.eye(r), "B'A^-1B")
    cov = sigma2 * Minv
    diag = Diagnostics(rank_A, dim_A, float(np.linalg.cond(s.A)), float(np.linalg.cond(M)), p, m, q, lay.absorbed,
                       first_stage_diagnostics(s))
    return FedMRResult(theta, cov, list(lay.w_names), s.N, rss, sigma2, df_resid, diag)


def first_stage_diagnostics(s: Stats) -> dict:
    lay = s.layout
    exog_z = lay.z_index(lay.exogenous) if lay.exogenous else np.array([], dtype=int)
    m = len(lay.instruments)
    out = {}
    for x in lay.endogenous:
        wi = lay.w_index([x])[0]
        bx = s.B[:, wi]
        xx = s.D[wi, wi]
        rss_full = xx - bx @ _solve(s.A, bx, "Z'Z")
        if len(exog_z):
            Ae, be = s.A[np.ix_(exog_z, exog_z)], bx[exog_z]
            rss_red = xx - be @ _solve(Ae, be, "C'C")
        else:
            rss_red = xx
        df2 = s.N - s.A.shape[0] - lay.absorbed
        F = ((rss_red - rss_full) / m) / (rss_full / df2)
        out[x] = {"F": float(F), "partial_r2": float((rss_red - rss_full) / rss_red), "df1": m, "df2": int(df2),
                  "conditional": len(lay.endogenous) == 1}
    return out


def robust_cov(s: Stats, H: np.ndarray) -> np.ndarray:
    """HC0 sandwich from the round-2 sum H = sum_k Z_k' diag(u_k^2) Z_k."""
    AinvB = _solve(s.A, s.B, "Z'Z")
    M = s.B.T @ AinvB
    Minv = _solve(M, np.eye(M.shape[0]), "B'A^-1B")
    return Minv @ AinvB.T @ H @ AinvB @ Minv
