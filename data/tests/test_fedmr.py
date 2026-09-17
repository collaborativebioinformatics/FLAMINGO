"""FedMR must equal the pooled fits in federated_summary_mr.py to machine precision."""

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
import fedmr  # noqa: E402
from federated_summary_mr import pooled_2sls, quadratic_2sls  # noqa: E402

FED = HERE.parent / "simulated_data" / "federated"
CONTINUOUS = [d for d in ("linear", "quadratic", "ushape", "threshold") if (FED / d / "manifest.json").exists()]
TOL = 1e-10


def raw(sites):
    return [(s.G, s.X, s.Y) for s in sites]


@pytest.fixture(scope="module", params=CONTINUOUS)
def sites(request):
    return fedmr.load_sites(FED / request.param)


def test_linear_identity_with_pooled_2sls(sites):
    res = fedmr.fedmr(sites, instruments="site", basis="linear")
    pooled, pooled_se, _ = pooled_2sls(raw(sites))
    assert abs(res["X"] - pooled) < TOL
    assert abs(res.se("X") - pooled_se) < TOL


def test_quadratic_identity_with_quadratic_2sls(sites):
    res = fedmr.fedmr(sites, instruments="site", basis="quadratic")
    theta, cov = quadratic_2sls(raw(sites))
    assert abs(res["X"] - theta[0]) < TOL and abs(res["X2"] - theta[1]) < TOL
    assert abs(res.se("X") - np.sqrt(cov[0, 0])) < TOL and abs(res.se("X2") - np.sqrt(cov[1, 1])) < TOL


def test_robust_cov_equals_stacked_hc0(sites):
    res = fedmr.fedmr(sites, instruments="site", basis="linear", robust=True)
    designs = [fedmr.build_design(s, "site", "linear") for s in sites]
    stats = fedmr.aggregate([fedmr.stats_from_design(s.name, d) for s, d in zip(sites, designs)])
    zi = {nm: i for i, nm in enumerate(stats.z_names)}
    wi = {nm: i for i, nm in enumerate(stats.w_names)}
    Z = np.zeros((stats.N, len(stats.z_names)))
    W = np.zeros((stats.N, len(stats.w_names)))
    Y = np.concatenate([d.Y for d in designs])
    row = 0
    for d in designs:
        Z[row:row + len(d.Y), [zi[n] for n in d.z_names]] = d.Z
        W[row:row + len(d.Y), [wi[n] for n in d.w_names]] = d.W
        row += len(d.Y)
    P = Z @ np.linalg.solve(Z.T @ Z, Z.T @ W)          # projected regressors
    theta = np.linalg.solve(P.T @ P, P.T @ Y)
    u = Y - W @ theta
    bread = np.linalg.inv(P.T @ P)
    V = bread @ (P.T @ (P * (u**2)[:, None])) @ bread
    assert np.allclose(res.theta, theta, atol=TOL)
    assert np.allclose(res.robust_cov, V, atol=TOL)


def test_partial_f_equals_nested_regression(sites):
    designs = [fedmr.build_design(s, "site", "linear") for s in sites]
    stats = fedmr.aggregate([fedmr.stats_from_design(s.name, d) for s, d in zip(sites, designs)])
    diag = fedmr.first_stage_diagnostics(stats)["X"]
    # direct: X on [site consts, site SNPs] versus X on site consts only
    X = np.concatenate([s.X for s in sites])
    K = len(sites)
    S = np.zeros((len(X), K))
    G = np.zeros((len(X), K * sites[0].G.shape[1]))
    row = 0
    for k, s in enumerate(sites):
        S[row:row + s.n, k] = 1
        G[row:row + s.n, k * s.G.shape[1]:(k + 1) * s.G.shape[1]] = s.G
        row += s.n
    rss = lambda Z: float(np.sum((X - Z @ np.linalg.lstsq(Z, X, rcond=None)[0]) ** 2))
    rss_r, rss_f = rss(S), rss(np.column_stack([S, G]))
    m, df2 = G.shape[1], len(X) - S.shape[1] - G.shape[1]
    assert abs(diag["F"] - ((rss_r - rss_f) / m) / (rss_f / df2)) < 1e-8
    assert abs(diag["partial_r2"] - (rss_r - rss_f) / rss_r) < TOL


def test_invariant_to_site_order_and_splitting(sites):
    a = fedmr.fedmr(sites, robust=True)
    b = fedmr.fedmr(sites[::-1], robust=True)
    assert abs(a["X"] - b["X"]) < TOL and abs(a.se("X", True) - b.se("X", True)) < TOL
    # splitting site01 into two halves that keep their own name-prefixed instrument columns
    # is a different model (two first stages); splitting under *shared* names must be invariant
    s0 = sites[0]
    h = s0.n // 2
    half1 = fedmr.SiteData(s0.name, s0.G[:h], s0.X[:h], s0.Y[:h])
    half2 = fedmr.SiteData(s0.name, s0.G[h:], s0.X[h:], s0.Y[h:])
    whole = fedmr.aggregate([fedmr.site_stats(s0)])
    split = fedmr.aggregate([fedmr.site_stats(half1), fedmr.site_stats(half2)])
    assert np.allclose(whole.A, split.A, atol=TOL) and np.allclose(whole.c, split.c, atol=TOL)
    assert abs(fedmr.fit(whole)["X"] - fedmr.fit(split)["X"]) < TOL


def test_shared_layout_equals_stacked_2sls():
    """With shared SNP names, FedMR must equal 2SLS on the stacked data with one G and site dummies."""
    rng = np.random.default_rng(0)
    m, sites = 5, []
    for k in range(3):
        n = 400 + 100 * k
        G = rng.binomial(2, 0.3, size=(n, m)).astype(float)
        U = rng.normal(size=n)
        X = G @ np.full(m, 0.2) + 0.5 * U + rng.normal(size=n)
        Y = 0.3 * X + 0.5 * U + rng.normal(size=n) + k
        sites.append(fedmr.SiteData(f"site{k}", G, X, Y))
    res = fedmr.fedmr(sites, instruments="shared", basis="linear")
    G = np.vstack([s.G for s in sites]); X = np.concatenate([s.X for s in sites]); Y = np.concatenate([s.Y for s in sites])
    S = np.zeros((len(X), 3)); row = 0
    for k, s in enumerate(sites):
        S[row:row + s.n, k] = 1; row += s.n
    Z, W = np.column_stack([G, S]), np.column_stack([X, S])
    P = Z @ np.linalg.lstsq(Z, W, rcond=None)[0]
    theta = np.linalg.lstsq(P, Y, rcond=None)[0]
    assert abs(res["X"] - theta[0]) < TOL


def test_transport_roundtrip(sites):
    st = fedmr.site_stats(sites[0])
    back = fedmr.SiteStats.from_transport(st.arrays(), st.meta())
    assert back.n == st.n and back.z_names == st.z_names and np.array_equal(back.A, st.A) and back.f == st.f


def test_oracle_column_rejected():
    with pytest.raises(ValueError):
        fedmr.SiteData("s", np.zeros((3, 1)), np.zeros(3), np.zeros(3), np.zeros((3, 1)), ("U",))


def test_crossfit_runs_and_is_close(sites):
    plain = fedmr.fedmr(sites)
    cf = fedmr.fedmr(sites, crossfit=5)
    assert abs(cf["X"] - plain["X"]) < 0.1
    assert np.isfinite(cf.se("X")) and cf.se("X") > 0
