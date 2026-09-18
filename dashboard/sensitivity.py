"""Sensitivity, robustness and invariance analyses from per-site sufficient statistics.

Model at site e (the same one federated_summary_mr.py fits):

    Y = alpha_e + phi(X)' theta + eps,   E_e[eps] = 0,   E_e[psi_e eps] = 0

with phi(X) = X (linear) or (X, X^2) (quadratic), site intercepts alpha_e as
nuisance parameters and instruments psi_e that are either the site's own SNPs
or the SNP-predicted exposure (xhat, xhat^2). Everything below is computed from
one small statistics bundle per site — n, the column means and the site-centred
cross-product matrix of (Y, X, X^2, U, xhat, xhat^2, yhat), plus G'G and G'(Y, X, X^2)
for the per-SNP analyses — so that every leave-one-out, subset, K-class, anchor,
minimax and J-test estimate is a sum of per-site matrices and never needs the
rows again. That is what a federation would exchange.

The oracle confounder U is carried only for the "classic ICP" comparison and is
labelled as an oracle wherever it appears.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
from scipy import optimize, stats as sps

COLS = ("Y", "X", "X2", "U", "xhat", "xhat2", "yhat")
IDX = {c: i for i, c in enumerate(COLS)}
SUPPORTED_SHAPES = ("linear", "quadratic", "threshold")

MODELS = {
    # regressors phi(X), and the instrument basis psi built from the first stage
    "linear": {"w": ("X",), "z": ("xhat",), "names": ("θ",)},
    "quadratic": {"w": ("X", "X2"), "z": ("xhat", "xhat2"), "names": ("θ1", "θ2")},
}


# --------------------------------------------------------------------------- data


@dataclass
class Perturbation:
    """What-if changes applied in memory to the loaded rows; nothing on disk changes.

    pleio_snps / pleio_alpha: SNP indices given a direct effect alpha on Y (a
        pleiotropic, invalid instrument) at the sites in pleio_sites ("all" or ids).
    deviant_site / deviant_delta: one site whose causal slope is theta + delta.
    shift_site / shift_c: one site whose outcome level is shifted by c (a direct
        site effect absorbed by the intercept).
    confound_site / confound_c: one site with extra confounding c * U on Y.
    """

    pleio_snps: tuple[int, ...] = ()
    pleio_alpha: float = 0.0
    pleio_sites: tuple[str, ...] | str = "all"
    deviant_site: str | None = None
    deviant_delta: float = 0.0
    shift_site: str | None = None
    shift_c: float = 0.0
    confound_site: str | None = None
    confound_c: float = 0.0

    def key(self) -> str:
        d = self.__dict__.copy()
        d["pleio_snps"] = list(self.pleio_snps)
        d["pleio_sites"] = self.pleio_sites if isinstance(self.pleio_sites, str) else list(self.pleio_sites)
        return json.dumps(d, sort_keys=True)

    def active(self) -> bool:
        return bool((self.pleio_snps and self.pleio_alpha) or (self.deviant_site and self.deviant_delta)
                    or (self.shift_site and self.shift_c) or (self.confound_site and self.confound_c))


@dataclass
class SiteStats:
    site: str
    n: int
    n_snps: int
    mean: np.ndarray          # (7,) means of COLS
    C: np.ndarray             # (7, 7) site-centred cross products of COLS
    GtG: np.ndarray           # (L, L) centred G'G
    GtW: np.ndarray           # (L, 3) centred G'(Y, X, X^2)
    truth: dict = field(default_factory=dict)

    @property
    def mean_F(self) -> float:
        """Mean per-SNP first-stage F, as in the summary-MR table."""
        bx, se = self.gwas("X")
        return float(np.mean((bx / se) ** 2))

    def gwas(self, col: str) -> tuple[np.ndarray, np.ndarray]:
        """Per-SNP simple regression of `col` in (Y, X, X2) on each SNP: (beta, se)."""
        j = ("Y", "X", "X2").index(col)
        sxx = np.diag(self.GtG)
        beta = self.GtW[:, j] / sxx
        rss = self.C[IDX[col], IDX[col]] - beta**2 * sxx
        se = np.sqrt(np.maximum(rss, 0) / (self.n - 2) / sxx)
        return beta, se


def _apply(pert: Perturbation, site: str, G: np.ndarray, X: np.ndarray, U: np.ndarray, Y: np.ndarray) -> np.ndarray:
    Y = Y.copy()
    if pert.pleio_snps and pert.pleio_alpha and (pert.pleio_sites == "all" or site in pert.pleio_sites):
        for j in pert.pleio_snps:
            if j < G.shape[1]:
                Y += pert.pleio_alpha * (G[:, j] - G[:, j].mean())
    if pert.deviant_site == site and pert.deviant_delta:
        Y += pert.deviant_delta * X
    if pert.shift_site == site and pert.shift_c:
        Y += pert.shift_c
    if pert.confound_site == site and pert.confound_c:
        Y += pert.confound_c * U
    return Y


def site_stats(site: str, G: np.ndarray, X: np.ndarray, Y: np.ndarray, U: np.ndarray, truth: dict) -> SiteStats:
    n = len(X)
    Gc = G - G.mean(axis=0)
    W = np.column_stack([Y, X, X**2])
    Wc = W - W.mean(axis=0)
    coef = np.linalg.lstsq(Gc, Wc, rcond=None)[0]        # reduced forms of Y, X, X^2 on the site's SNPs
    fitted = Gc @ coef + W.mean(axis=0)
    xhat, yhat = fitted[:, 1], fitted[:, 0]
    V = np.column_stack([Y, X, X**2, U, xhat, xhat**2, yhat])
    Vc = V - V.mean(axis=0)
    return SiteStats(site=site, n=n, n_snps=G.shape[1], mean=V.mean(axis=0), C=Vc.T @ Vc,
                     GtG=Gc.T @ Gc, GtW=Gc.T @ Wc, truth=truth)


def load_sites(folder: Path, pert: Perturbation | None = None) -> list[SiteStats]:
    """Per-site sufficient statistics for every site*.csv in `folder` (linear/quadratic/threshold data)."""
    pert = pert or Perturbation()
    out = []
    for csv in sorted(Path(folder).glob("site*.csv")):
        df = pl.read_csv(csv)
        if "Y" not in df.columns:
            raise ValueError(f"{csv.name} has no Y column; the sufficient-statistics analyses need a continuous outcome")
        G = df.select(pl.col("^snp.*$")).to_numpy().astype(float)
        X, Y = df["X"].to_numpy().astype(float), df["Y"].to_numpy().astype(float)
        U = df["U"].to_numpy().astype(float) if "U" in df.columns else np.zeros(len(X))
        truth_path = csv.with_suffix(".truth.json")
        truth = json.loads(truth_path.read_text()) if truth_path.exists() else {}
        truth = {k: truth.get(k) for k in ("avg_slope", "h2_x", "gamma_x", "gamma_y", "theta1", "theta2", "theta")}
        Y = _apply(pert, csv.stem, G, X, U, Y)
        out.append(site_stats(csv.stem, G, X, Y, U, truth))
    return out


def truth_for(sites: list[SiteStats], manifest: dict, model: str) -> np.ndarray | None:
    """The parameters a `model` fit targets: (theta1, theta2) for a quadratic truth, the
    n-weighted average slope for a linear fit; None where no truth is defined (threshold θ2)."""
    shape = manifest.get("shape") or ("quadratic" if manifest.get("theta2") is not None else "linear")
    if model == "linear":
        n = np.array([s.n for s in sites], float)
        slopes = np.array([s.truth.get("avg_slope") if s.truth.get("avg_slope") is not None else manifest.get("theta1", np.nan)
                           for s in sites], float)
        return np.array([float(np.sum(n * slopes) / n.sum())])
    if shape == "quadratic":
        return np.array([manifest.get("theta1", np.nan), manifest.get("theta2", np.nan)], float)
    return np.array([manifest.get("theta1", np.nan), np.nan])


# ----------------------------------------------------------------- moment algebra


def _solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(A) @ b


def _inv(A: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(A)


def moments(s: SiteStats, model: str, instr: str, snps: np.ndarray | None = None):
    """Site-centred (Q = Z'Z, D = Z'W, a = Z'Y) for the model's regressors W and instruments Z.

    instr = "basis": Z is (xhat) or (xhat, xhat^2), just-identified.
    instr = "snps":  Z is the site's SNPs (optionally the subset `snps`), over-identified.
    """
    m = MODELS[model]
    wi = [IDX[c] for c in m["w"]]
    if instr == "basis":
        zi = [IDX[c] for c in m["z"]]
        return s.C[np.ix_(zi, zi)], s.C[np.ix_(zi, wi)], s.C[zi, IDX["Y"]]
    j = np.arange(s.n_snps) if snps is None else np.asarray(snps)
    wcols = [("Y", "X", "X2").index(c) for c in m["w"]]
    return s.GtG[np.ix_(j, j)], s.GtW[np.ix_(j, wcols)], s.GtW[j, 0]


@dataclass
class Agg:
    """Sufficient statistics of a set of sites, all with site intercepts partialled out."""

    n: int
    K: int
    q: int                    # total number of instruments across sites
    p: int
    Cyy: float
    Cwy: np.ndarray
    Cww: np.ndarray
    yPy: float                # Y' P_Z Y with block-diagonal (per-site) projection
    WPy: np.ndarray
    WPW: np.ndarray

    def rss(self, theta: np.ndarray) -> float:
        return float(self.Cyy - 2 * theta @ self.Cwy + theta @ self.Cww @ theta)

    def rPr(self, theta: np.ndarray) -> float:
        return float(self.yPy - 2 * theta @ self.WPy + theta @ self.WPW @ theta)


def aggregate(sites: list[SiteStats], model: str, instr: str = "basis", snps: np.ndarray | None = None,
              pooled: bool | None = None) -> Agg:
    """Sum the per-site moments.

    pooled=True adds the moment conditions across sites (one Z'Z, Z'W, Z'Y for the whole
    federation), which is the pooled 2SLS of federated_summary_mr.py; it is the right thing
    for the (xhat, xhat^2) basis, whose columns mean the same at every site. pooled=False
    keeps one block of instruments per site (block-diagonal projection), which is what SNP
    instruments need since SNPs are site-specific draws. Default: pooled for the basis.
    """
    m = MODELS[model]
    wi = [IDX[c] for c in m["w"]]
    p = len(wi)
    if pooled is None:
        pooled = instr == "basis"
    n = K = q = 0
    Cyy, Cwy, Cww = 0.0, np.zeros(p), np.zeros((p, p))
    yPy, WPy, WPW = 0.0, np.zeros(p), np.zeros((p, p))
    Qs, Ds, As = 0.0, 0.0, 0.0
    for s in sites:
        Q, D, a = moments(s, model, instr, snps)
        n += s.n; K += 1; q += Q.shape[0]
        Cyy += s.C[IDX["Y"], IDX["Y"]]; Cwy += s.C[wi, IDX["Y"]]; Cww += s.C[np.ix_(wi, wi)]
        if pooled:
            Qs = Qs + Q; Ds = Ds + D; As = As + a
        else:
            Qi = _inv(Q)
            yPy += a @ Qi @ a; WPy += D.T @ Qi @ a; WPW += D.T @ Qi @ D
    if pooled:
        Qi = _inv(Qs)
        yPy, WPy, WPW = As @ Qi @ As, Ds.T @ Qi @ As, Ds.T @ Qi @ Ds
        q = Qs.shape[0]
    return Agg(n, K, q, p, float(Cyy), Cwy, Cww, float(yPy), WPy, WPW)


@dataclass
class Estimate:
    theta: np.ndarray
    cov: np.ndarray
    label: str = ""

    @property
    def se(self) -> np.ndarray:
        return np.sqrt(np.maximum(np.diag(self.cov), 0))


def two_sls(A: Agg, label: str = "2SLS") -> Estimate:
    theta = _solve(A.WPW, A.WPy)
    sigma2 = A.rss(theta) / max(A.n - A.K - A.p, 1)
    return Estimate(theta, sigma2 * _inv(A.WPW), label)


def ols(A: Agg, label: str = "OLS (no instruments)") -> Estimate:
    theta = _solve(A.Cww, A.Cwy)
    sigma2 = A.rss(theta) / max(A.n - A.K - A.p, 1)
    return Estimate(theta, sigma2 * _inv(A.Cww), label)


def kclass(A: Agg, kappa: float) -> np.ndarray:
    """K-class estimator: kappa = 0 is OLS, 1 is 2SLS, kappa_LIML gives LIML."""
    return _solve((1 - kappa) * A.Cww + kappa * A.WPW, (1 - kappa) * A.Cwy + kappa * A.WPy)


def liml_kappa(A: Agg) -> float:
    Om = np.block([[np.array([[A.Cyy]]), A.Cwy[None, :]], [A.Cwy[:, None], A.Cww]])
    OmP = np.block([[np.array([[A.yPy]]), A.WPy[None, :]], [A.WPy[:, None], A.WPW]])
    ev = np.linalg.eigvals(_solve(Om - OmP, Om))
    return float(np.min(ev.real))


def pulse_stat(A: Agg, theta: np.ndarray) -> float:
    """n * R^2 of the residual on the (per-site) instruments: the test statistic behind
    the PULSE (Jakobsen & Peters 2022), chi^2 with q degrees of freedom under validity."""
    return float(A.n * A.rPr(theta) / A.rss(theta))


def kclass_path(A: Agg, alpha: float = 0.05, n_grid: int = 241) -> dict:
    """OLS -> 2SLS -> LIML path with the PULSE stopping point.

    The PULSE takes the smallest kappa (closest to OLS) whose residuals pass the
    instrument test at level alpha; if even 2SLS fails, it reports 2SLS and flags it.
    """
    k_liml = liml_kappa(A)
    k_fuller = k_liml - 1.0 / max(A.n - A.q - A.K, 1)
    # the K-class matrix turns singular a little beyond kappa_LIML, so the path ends at LIML
    k_max = max(1.0, k_liml)
    kappas = np.linspace(0.0, k_max, n_grid)
    thetas = np.array([kclass(A, k) for k in kappas])
    T = np.array([pulse_stat(A, t) for t in thetas])
    crit = sps.chi2.ppf(1 - alpha, A.q)
    ok = np.where((T <= crit) & (kappas <= 1.0))[0]
    if len(ok):
        k_pulse, pulse_rejected = float(kappas[ok[0]]), False
    else:
        k_pulse, pulse_rejected = 1.0, True
    marks = {"OLS": 0.0, "PULSE": k_pulse, "2SLS": 1.0, "Fuller(1)": k_fuller, "LIML": k_liml}
    return {"kappa": kappas, "theta": thetas, "T": T, "crit": crit, "df": A.q,
            "marks": {k: (v, kclass(A, v), pulse_stat(A, kclass(A, v))) for k, v in marks.items()},
            "pulse_rejected": pulse_rejected, "alpha": alpha}


def ridge_path(A: Agg, lambdas: np.ndarray) -> np.ndarray:
    """Ridge-penalised 2SLS: argmin ||P_Z (Y - W theta)||^2 + lambda * tr(W'P_Z W)/p * ||theta||^2.
    lambda is relative to the average instrument signal so the path looks alike across data sizes."""
    scale = np.trace(A.WPW) / A.p
    return np.array([_solve(A.WPW + lam * scale * np.eye(A.p), A.WPy) for lam in lambdas])


# -------------------------------------------------------- per-site and meta-analysis


def per_site(sites: list[SiteStats], model: str, instr: str = "basis") -> list[Estimate]:
    return [two_sls(aggregate([s], model, instr), s.site) for s in sites]


def per_site_ols(sites: list[SiteStats], model: str) -> list[Estimate]:
    return [ols(aggregate([s], model), s.site) for s in sites]


def ivw_meta(ests: list[Estimate], power: float = 1.0) -> Estimate:
    """Precision-weighted combination of vector estimates. power = 1 is inverse-variance
    weighting (the fixed-effect meta-analysis); power = 0 weights every site equally."""
    W = [np.linalg.matrix_power(_inv(e.cov), 1) for e in ests]
    if power != 1.0:
        # scale each site's precision by (its trace / mean trace)^(power - 1) so that
        # power = 0 removes size differences while keeping the shape of the covariances
        tr = np.array([np.trace(w) for w in W]); tr_mean = tr.mean()
        W = [w * (tr_mean / t) ** (1 - power) for w, t in zip(W, tr)]
    cov = _inv(sum(W))
    theta = cov @ sum(w @ e.theta for w, e in zip(W, ests))
    return Estimate(theta, cov, f"IVW (power {power:g})")


def heterogeneity(ests: list[Estimate], component: int = 0) -> dict:
    """DerSimonian-Laird random effects for one component: Q, I^2, tau^2, RE estimate, prediction interval."""
    y = np.array([e.theta[component] for e in ests]); v = np.array([e.cov[component, component] for e in ests])
    w = 1 / v
    fe = np.sum(w * y) / np.sum(w); fe_se = np.sqrt(1 / np.sum(w))
    Q = float(np.sum(w * (y - fe) ** 2)); df = len(y) - 1
    tau2 = max(0.0, (Q - df) / (np.sum(w) - np.sum(w**2) / np.sum(w))) if df > 0 else 0.0
    wr = 1 / (v + tau2)
    re = np.sum(wr * y) / np.sum(wr); re_se = np.sqrt(1 / np.sum(wr))
    I2 = max(0.0, (Q - df) / Q) if Q > 0 else 0.0
    t = sps.t.ppf(0.975, df - 1) if df > 1 else 1.96
    pi = np.sqrt(tau2 + re_se**2)
    return {"fe": fe, "fe_se": fe_se, "re": re, "re_se": re_se, "Q": Q, "df": df,
            "p": float(sps.chi2.sf(Q, df)) if df > 0 else np.nan, "I2": I2, "tau2": tau2, "tau": np.sqrt(tau2),
            "pi_lo": re - t * pi, "pi_hi": re + t * pi}


def pairwise_z(ests: list[Estimate], component: int = 0) -> np.ndarray:
    y = np.array([e.theta[component] for e in ests]); v = np.array([e.cov[component, component] for e in ests])
    return (y[:, None] - y[None, :]) / np.sqrt(v[:, None] + v[None, :])


def site_disagreement(sites: list[SiteStats], model: str, instr: str = "basis", scale: str = "joint") -> dict:
    """Disagreement z-scores from the joint 2SLS model: site intercepts, one slope vector per site,
    one instrument block per site and (scale="joint") a single error variance for the federation.

    The joint model's estimate of the common slope is the precision-weighted mean of the site slopes
    with precisions W_e'P_e W_e / sigma^2, and
      z_joint[e]    compares site e with the joint fit of every *other* site: the Wald z of the
                    site x X interaction. The two fits use disjoint rows, so their variances add.
      z_pair[e, f]  compares two sites on the same noise scale (row minus column).
      Q             tests that every site shares the joint slope, chi^2 on (K - 1) p df.
    scale="site" uses each site's own residual variance instead, which is the meta-analysis view:
    a site whose noise is inflated then buys itself a smaller z.
    """
    aggs = [aggregate([s], model, instr) for s in sites]
    thetas = np.array([_solve(A.WPW, A.WPy) for A in aggs])
    K, p = thetas.shape
    n = sum(A.n for A in aggs)
    sigma2_joint = sum(A.rss(t) for A, t in zip(aggs, thetas)) / max(n - K - K * p, 1)
    sigma2_site = np.array([A.rss(t) / max(A.n - 1 - p, 1) for A, t in zip(aggs, thetas)])
    s2 = sigma2_site if scale == "site" else np.full(K, sigma2_joint)
    P = [A.WPW / v for A, v in zip(aggs, s2)]                       # per-site precisions on the chosen scale
    var = np.array([np.diag(_inv(Pe)) for Pe in P])                 # K x p
    z_pair = (thetas[:, None, :] - thetas[None, :, :]) / np.sqrt(var[:, None, :] + var[None, :, :])
    z_joint = np.full((K, p), np.nan)
    if K >= 2:
        for e in range(K):
            Pr = sum(P[f] for f in range(K) if f != e)
            Vr = _inv(Pr)
            tr = Vr @ sum(P[f] @ thetas[f] for f in range(K) if f != e)
            z_joint[e] = (thetas[e] - tr) / np.sqrt(var[e] + np.diag(Vr))
    Vj = _inv(sum(P)); theta_joint = Vj @ sum(Pe @ t for Pe, t in zip(P, thetas))
    Q = float(sum((t - theta_joint) @ Pe @ (t - theta_joint) for Pe, t in zip(P, thetas)))
    df = (K - 1) * p
    return {"theta": thetas, "theta_joint": theta_joint, "se_joint": np.sqrt(np.diag(Vj)), "z_pair": z_pair, "z_joint": z_joint,
            "sigma2": sigma2_joint, "sigma2_site": sigma2_site, "Q": Q, "df": df,
            "p": float(sps.chi2.sf(Q, df)) if df > 0 else np.nan}


# ------------------------------------------------------------- subsets of sites


def leave_out(sites: list[SiteStats], model: str, k: int = 1, instr: str = "basis",
              estimator=two_sls, max_subsets: int = 400, seed: int = 0) -> list[dict]:
    """Estimates with every subset of k sites removed (random subsets beyond max_subsets)."""
    K = len(sites)
    combos = list(itertools.combinations(range(K), k))
    if len(combos) > max_subsets:
        rng = np.random.default_rng(seed)
        combos = [combos[i] for i in rng.choice(len(combos), max_subsets, replace=False)]
    out = []
    for drop in combos:
        keep = [s for i, s in enumerate(sites) if i not in drop]
        est = estimator(aggregate(keep, model, instr))
        out.append({"dropped": tuple(sites[i].site for i in drop), "theta": est.theta, "se": est.se,
                    "n": sum(s.n for s in keep)})
    return out


def cumulative(sites: list[SiteStats], model: str, order: list[int], instr: str = "basis") -> list[Estimate]:
    return [two_sls(aggregate([sites[i] for i in order[: j + 1]], model, instr), sites[order[j]].site)
            for j in range(len(order))]


# ------------------------------------------------- robustness across sites (anchors, DRO)


def anchor_path(sites: list[SiteStats], model: str, gammas: np.ndarray, iv: bool = True) -> np.ndarray:
    """Anchor regression with the site indicator as anchor (Rothenhäusler et al. 2021).

    argmin_theta  ||(I - P_A) r||^2 + gamma ||P_A r||^2,  r = Y - c - R theta,  A = site dummies,

    so gamma = 0 is the site-fixed-effects fit, gamma = 1 the pooled fit with one
    intercept and gamma -> inf the between-site regression of site means. With
    iv=True the regressors R are the first-stage fitted (xhat, xhat^2): the anchored
    second stage of 2SLS. Returns theta(gamma) as (len(gammas), p).
    """
    m = MODELS[model]
    ri = [IDX[c] for c in (m["z"] if iv else m["w"])]
    yi = IDX["Y"]
    n = sum(s.n for s in sites)
    mbar = sum(s.n * s.mean for s in sites) / n
    Cw = sum(s.C for s in sites)
    B = sum(s.n * np.outer(s.mean - mbar, s.mean - mbar) for s in sites)
    out = []
    for g in gammas:
        M = Cw + g * B
        out.append(_solve(M[np.ix_(ri, ri)], M[ri, yi]) if np.isfinite(g) else _solve(B[np.ix_(ri, ri)], B[ri, yi]))
    return np.array(out)


def wald_surprise(ests: list[Estimate], theta: np.ndarray) -> np.ndarray:
    """Each site's Wald statistic for the candidate theta: (theta - theta_e)' V_e^-1 (theta - theta_e)."""
    return np.array([(theta - e.theta) @ _inv(e.cov) @ (theta - e.theta) for e in ests])


def minimax(ests: list[Estimate], start: np.ndarray) -> np.ndarray:
    """The theta that minimises the largest site Wald statistic: the Chebyshev centre of the
    site confidence ellipsoids, i.e. group-DRO with each site's own scale (no site is very surprised)."""
    f = lambda t: float(np.max(wald_surprise(ests, np.atleast_1d(t))))
    if len(start) == 1:
        lo = min(e.theta[0] - 4 * e.se[0] for e in ests); hi = max(e.theta[0] + 4 * e.se[0] for e in ests)
        grid = np.linspace(lo, hi, 4001)
        vals = np.max([((grid - e.theta[0]) / e.se[0]) ** 2 for e in ests], axis=0)
        return np.array([grid[np.argmin(vals)]])
    res = optimize.minimize(f, start, method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 4000})
    return res.x


def excess_risk(sites: list[SiteStats], model: str, ests: list[Estimate], theta: np.ndarray, instr: str = "basis") -> np.ndarray:
    """Per-site second-stage excess risk (theta - theta_e)' W_e'P W_e (theta - theta_e) / n_e, which
    removes each site's own noise level so that equalising it targets the fit, not the noise."""
    out = []
    for s, e in zip(sites, ests):
        A = aggregate([s], model, instr)
        d = theta - e.theta
        out.append(float(d @ A.WPW @ d) / s.n)
    return np.array(out)


def vrex_path(sites: list[SiteStats], model: str, ests: list[Estimate], start: np.ndarray, lambdas: np.ndarray,
              instr: str = "basis") -> np.ndarray:
    """V-REx (Krueger et al. 2021) on the excess risks: sum_e pi_e R_e + lambda * Var_pi(R_e) with
    pi_e = n_e / n, so that lambda -> 0 is the (stacked) pooled second stage."""
    out, x0 = [], start.copy()
    pi = np.array([s.n for s in sites], float); pi /= pi.sum()
    for lam in lambdas:
        def obj(t):
            r = excess_risk(sites, model, ests, np.atleast_1d(t), instr)
            mean = float(pi @ r)
            return mean + lam * float(pi @ (r - mean) ** 2)
        res = optimize.minimize(obj, x0, method="Nelder-Mead", options={"xatol": 1e-7, "fatol": 1e-12, "maxiter": 4000})
        x0 = res.x
        out.append(res.x.copy())
    return np.array(out)


# ------------------------------------------------------- invariance certificate (GMM J)


def certificate(sites: list[SiteStats], model: str, instr: str = "snps", snps: np.ndarray | None = None) -> dict:
    """Two-step GMM on the stacked per-site IV moments with homoskedastic per-site weights.

    J = sum_e J_e (within: each site's own Sargan statistic, instrument validity)
      + sum_e (theta_e - theta)' V_e^-1 (theta_e - theta)   (between: one theta for all sites),

    exact because theta is the V_e-weighted mean of the site estimates theta_e.
    """
    p = len(MODELS[model]["w"])
    rows, Ws, thetas = [], [], []
    for s in sites:
        Q, D, a = moments(s, model, instr, snps)
        A = aggregate([s], model, instr, snps)
        e = two_sls(A)
        sigma2 = A.rss(e.theta) / max(s.n - 1 - p, 1)
        J_e = A.rPr(e.theta) / sigma2                      # residual on instruments at the site's own theta
        Vinv = D.T @ _inv(Q) @ D / sigma2
        rows.append({"site": s.site, "J": float(J_e), "df": Q.shape[0] - p, "theta": e.theta, "Vinv": Vinv, "n": s.n})
        Ws.append(Vinv); thetas.append(e.theta)
    cov = _inv(sum(Ws)); theta = cov @ sum(w @ t for w, t in zip(Ws, thetas))
    within = sum(r["J"] for r in rows); df_within = sum(r["df"] for r in rows)
    between = float(sum((r["theta"] - theta) @ r["Vinv"] @ (r["theta"] - theta) for r in rows))
    df_between = (len(sites) - 1) * p
    for r in rows:
        r["p"] = float(sps.chi2.sf(r["J"], r["df"])) if r["df"] > 0 else np.nan
        r["between"] = float((r["theta"] - theta) @ r["Vinv"] @ (r["theta"] - theta))
    return {"theta": theta, "cov": cov, "within": within, "df_within": df_within,
            "p_within": float(sps.chi2.sf(within, df_within)) if df_within > 0 else np.nan,
            "between": between, "df_between": df_between,
            "p_between": float(sps.chi2.sf(between, df_between)) if df_between > 0 else np.nan,
            "J": within + between, "df": df_within + df_between,
            "p": float(sps.chi2.sf(within + between, df_within + df_between)), "sites": rows}


# -------------------------------------------------- ICP-style invariant instrument search


def snp_invariance(sites: list[SiteStats], theta: float) -> dict:
    """Per-SNP, per-site test of the ratio invariance beta_y = theta * beta_x.

    A valid instrument has the same causal ratio at every site; a pleiotropic one with
    direct effect alpha has ratio theta + alpha / beta_x, which moves with the site's
    beta_x. z has shape (sites, snps); T_j = sum_e z_ej^2 ~ chi^2(K) for a valid SNP j.
    """
    L = min(s.n_snps for s in sites)
    z = np.zeros((len(sites), L))
    for i, s in enumerate(sites):
        bx, sx = s.gwas("X"); by, sy = s.gwas("Y")
        z[i] = (by[:L] - theta * bx[:L]) / np.sqrt(sy[:L] ** 2 + theta**2 * sx[:L] ** 2)
    T = (z**2).sum(axis=0)
    return {"z": z, "T": T, "df": len(sites), "p": sps.chi2.sf(T, len(sites)),
            "crit": sps.chi2.ppf(0.95, len(sites))}


def invariant_instrument_search(sites: list[SiteStats], alpha: float = 0.05, max_steps: int | None = None) -> dict:
    """Greedy backward search for the largest SNP set S whose stacked-moment J test (with
    Z = G_S at every site) is not rejected: the ICP-style "accepted set" for instruments.

    ICP intersects the accepted sets to find the parents. For instruments any subset of a
    valid set stays valid, so the informative object is the largest accepted set, and the
    SNPs removed on the way are the ones the sites disagree about.
    """
    L = min(s.n_snps for s in sites)
    K = len(sites)
    keep = list(range(L))
    path = []
    steps = 0
    accepted = False
    reason = ""
    # a SNP is only removed when its own cross-site test rejects (Bonferroni over the SNPs
    # still in play): a between-site violation that no instrument causes cannot be fixed by
    # dropping instruments, and the search says so instead of emptying the set
    while len(keep) > 1:
        cert = certificate(sites, "linear", "snps", np.array(keep))
        theta = float(cert["theta"][0]); se = float(np.sqrt(cert["cov"][0, 0]))
        inv = snp_invariance(sites, theta)
        path.append({"removed": None if not path else path[-1]["candidate"], "size": len(keep), "theta": theta, "se": se,
                     "J": cert["J"], "df": cert["df"], "p": cert["p"], "p_within": cert["p_within"],
                     "p_between": cert["p_between"], "candidate": None})
        if cert["p"] > alpha:
            accepted = True
            break
        if max_steps is not None and steps >= max_steps:
            reason = "step limit reached"
            break
        T_keep = inv["T"][keep]
        worst = keep[int(np.argmax(T_keep))]
        if T_keep.max() < sps.chi2.ppf(1 - alpha / len(keep), K):
            reason = ("no single SNP disagrees across sites, yet the J test rejects: the violation is "
                      + ("between sites (θ differs by site)" if cert["p_between"] < alpha else "within sites (instruments jointly)"))
            break
        path[-1]["candidate"] = worst
        keep.remove(worst)
        steps += 1
    return {"accepted": keep, "is_accepted": accepted, "reason": reason,
            "removed": [p["candidate"] for p in path if p["candidate"] is not None], "path": path, "alpha": alpha}


# ------------------------------------------------------------ classic and IV ICP


def _total(sites: list[SiteStats]) -> tuple[np.ndarray, int]:
    """Cross products centred at the pooled mean (one global intercept), and n."""
    n = sum(s.n for s in sites)
    mbar = sum(s.n * s.mean for s in sites) / n
    T = sum(s.C + s.n * np.outer(s.mean - mbar, s.mean - mbar) for s in sites)
    return T, n


def _rss_ols(T: np.ndarray, si: list[int]) -> float:
    yi = IDX["Y"]
    if not si:
        return float(T[yi, yi])
    b = _solve(T[np.ix_(si, si)], T[si, yi])
    return float(T[yi, yi] - b @ T[si, yi])


def classic_icp(sites: list[SiteStats], candidates: tuple[str, ...] = ("X", "X2", "U"), alpha: float = 0.05) -> dict:
    """Invariant causal prediction (Peters, Bühlmann, Meinshausen 2016) with sites as environments,
    on the observed predictors and the oracle confounder U.

    For each candidate set S, Y is regressed on S with one intercept. The invariance of
    Y | X_S across sites is tested per site against all other sites with a Chow test on
    the coefficients and an F test on the residual variances; the set's p-value is the
    Bonferroni-corrected minimum. The estimated parents are the intersection of the
    accepted sets.
    """
    K = len(sites)
    sets = [()] + [c for r in range(1, len(candidates) + 1) for c in itertools.combinations(candidates, r)]
    results = []
    for S in sets:
        si = [IDX[c] for c in S]
        k = len(si) + 1
        T_all, n_all = _total(sites)
        rss_all = _rss_ols(T_all, si)
        per_site = []
        for i, s in enumerate(sites):
            rest = [t for j, t in enumerate(sites) if j != i]
            T_e, n_e = _total([s]); T_r, n_r = _total(rest)
            rss_e, rss_r = _rss_ols(T_e, si), _rss_ols(T_r, si)
            df2 = n_all - 2 * k
            F_chow = ((rss_all - rss_e - rss_r) / k) / ((rss_e + rss_r) / df2)
            p_chow = float(sps.f.sf(max(F_chow, 0), k, df2))
            s2_e, s2_r = rss_e / (n_e - k), rss_r / (n_r - k)
            F_var = s2_e / s2_r
            p_var = float(2 * min(sps.f.cdf(F_var, n_e - k, n_r - k), sps.f.sf(F_var, n_e - k, n_r - k)))
            per_site.append({"site": s.site, "p_chow": p_chow, "p_var": p_var, "p": min(1.0, 2 * min(p_chow, p_var))})
        p_set = min(1.0, K * min(r["p"] for r in per_site))
        results.append({"set": S, "p": p_set, "accepted": p_set > alpha, "sites": per_site})
    accepted = [set(r["set"]) for r in results if r["accepted"]]
    parents = set.intersection(*accepted) if accepted else None
    return {"results": results, "parents": parents, "alpha": alpha, "candidates": candidates}


def iv_icp(sites: list[SiteStats], alpha: float = 0.05) -> dict:
    """The IV counterpart: for S in {X}, {X, X^2}, fit the pooled fixed-effects 2SLS and test, per site,
    that the structural residual is orthogonal to the full instrument basis (xhat, xhat^2), i.e. that
    the site's IV moments hold at the shared theta. Hidden confounding is allowed to differ between
    sites; only the causal curve has to be invariant."""
    K = len(sites)
    zi = [IDX["xhat"], IDX["xhat2"]]
    results = []
    for model in ("linear", "quadratic"):
        wi = [IDX[c] for c in MODELS[model]["w"]]
        est = two_sls(aggregate(sites, model, "basis"))
        c = np.zeros(len(COLS)); c[IDX["Y"]] = 1; c[wi] -= est.theta          # residual = V c (site-centred)
        per_site = []
        for s in sites:
            Czz = s.C[np.ix_(zi, zi)]; Czr = s.C[zi] @ c; rr = c @ s.C @ c
            b = _solve(Czz, Czr)
            ss_expl = float(b @ Czz @ b)
            df2 = s.n - 1 - len(zi)
            F = (ss_expl / len(zi)) / max(rr - ss_expl, 1e-12) * df2
            per_site.append({"site": s.site, "F": F, "p": float(sps.f.sf(F, len(zi), df2))})
        p_set = min(1.0, K * min(r["p"] for r in per_site))
        results.append({"set": MODELS[model]["w"], "model": model, "theta": est.theta, "se": est.se,
                        "p": p_set, "accepted": p_set > alpha, "sites": per_site})
    accepted = [set(r["set"]) for r in results if r["accepted"]]
    parents = set.intersection(*accepted) if accepted else None
    return {"results": results, "parents": parents, "alpha": alpha}
