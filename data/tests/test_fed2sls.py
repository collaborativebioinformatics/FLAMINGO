"""Fed-2SLS must equal pooled 2SLS to machine precision.

The references are direct stacked-data projections written out here, not
another call through the package's matrix helpers, plus the repo's own
pooled_2sls / quadratic_2sls for the local-first-stage protocol.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

import flamingo_fedmr as fm
from flamingo_fedmr.protocols import design_generated, design_shared, local_first_stage

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
from federated_summary_mr import pooled_2sls, quadratic_2sls  # noqa: E402

FED = HERE.parent / "simulated_data" / "federated"
CONTINUOUS = [d for d in ("linear", "quadratic", "ushape", "threshold") if (FED / d / "manifest.json").exists()]
TOL = 1e-10


def raw(sites: list[fm.SiteData]) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    return [(s.G, s.X, s.Y) for s in sites]


def stacked_2sls(Z, W, Y, absorbed) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Reference 2SLS on stacked rows: projection, coefficients, classical and HC0 covariance."""
    P = Z @ np.linalg.lstsq(Z, W, rcond=None)[0]
    theta = np.linalg.lstsq(P, Y, rcond=None)[0]
    u = Y - W @ theta
    sigma2 = u @ u / (len(Y) - W.shape[1] - absorbed)
    bread = np.linalg.inv(P.T @ P)
    return theta, sigma2 * bread, bread @ (P.T @ (P * (u**2)[:, None])) @ bread, sigma2


def stack_with_dummies(sites, z_of, w_of) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack sites with explicit site dummies in both Z and W (the un-centred formulation)."""
    K = len(sites)
    Z, W, Y = [], [], []
    for k, s in enumerate(sites):
        S = np.zeros((s.n, K))
        S[:, k] = 1
        Z.append(np.column_stack([z_of(s), S]))
        W.append(np.column_stack([w_of(s), S]))
        Y.append(s.Y)
    return np.vstack(Z), np.vstack(W), np.concatenate(Y)


@pytest.fixture(scope="module", params=CONTINUOUS)
def sites(request) -> list[fm.SiteData]:
    return fm.load_sites(FED / request.param)


@pytest.fixture(scope="module")
def shared_sites() -> list[fm.SiteData]:
    """Small synthetic shared-SNP sites with one aligned covariate, built here so the test
    does not depend on a checked-in set."""
    rng = np.random.default_rng(0)
    m, out = 6, []
    for k in range(4):
        n = 300 + 150 * k
        G = rng.binomial(2, rng.uniform(0.1, 0.5, m), size=(n, m)).astype(float)
        age = rng.normal(50, 10, n)
        U = rng.normal(size=n)
        X = G @ np.linspace(0.1, 0.3, m) + 0.01 * age + 0.5 * U + rng.normal(size=n)
        Y = 0.3 * X + 0.02 * age + 0.5 * U + rng.normal(size=n) + 2 * k
        out.append(fm.SiteData(f"site{k}", G, X, Y, age[:, None], ("age",)))
    return out


# --------------------------------------------------------------------------- shared-instrument protocol


def test_shared_linear_equals_stacked_2sls_with_dummies(shared_sites) -> None:
    run = fm.SharedInstrumentFedMR(basis="linear", robust=True).run(shared_sites)
    res = run.result
    Z, W, Y = stack_with_dummies(shared_sites, lambda s: np.column_stack([s.G, s.C]),
                                 lambda s: np.column_stack([s.X, s.C]))
    theta, cov, hc0, sigma2 = stacked_2sls(Z, W, Y, absorbed=0)
    r = len(res.theta)                                   # X, cov:age; dummies come after
    assert np.allclose(res.theta, theta[:r], atol=TOL)
    assert np.allclose(res.cov, cov[:r, :r], atol=TOL)
    assert np.allclose(res.robust_cov, hc0[:r, :r], atol=TOL)
    assert abs(res.sigma2 - sigma2) < TOL
    assert run.rounds == 2 and res.diagnostics.absorbed == len(shared_sites)


def test_shared_first_stage_F_equals_nested_regression(shared_sites) -> None:
    run = fm.SharedInstrumentFedMR(basis="linear", robust=False).run(shared_sites)
    d = run.result.diagnostics.first_stage["X"]
    Z, W, Y = stack_with_dummies(shared_sites, lambda s: np.column_stack([s.G, s.C]),
                                 lambda s: np.column_stack([s.X, s.C]))
    X = W[:, 0]
    m = shared_sites[0].G.shape[1]
    exog = Z[:, m:]                                      # covariate + dummies
    def residual_sum_squares(design):
        residual = X - design @ np.linalg.lstsq(design, X, rcond=None)[0]
        return float(residual @ residual)

    rss_r, rss_f = residual_sum_squares(exog), residual_sum_squares(Z)
    df2 = len(X) - Z.shape[1]
    assert abs(d["F"] - ((rss_r - rss_f) / m) / (rss_f / df2)) < 1e-8
    assert abs(d["partial_r2"] - (rss_r - rss_f) / rss_r) < TOL
    assert d["conditional"] is True and d["df1"] == m and d["df2"] == df2


def test_shared_quadratic_equals_stacked_generated_instrument_2sls(shared_sites) -> None:
    run = fm.SharedInstrumentFedMR(basis="quadratic", robust=False).run(shared_sites)
    assert run.rounds == 2
    # reference: global first stage with dummies, then 2SLS on [xhat, xhat^2, C, S]
    Z1, _, _ = stack_with_dummies(shared_sites, lambda s: np.column_stack([s.G, s.C]), lambda s: s.X[:, None])
    X = np.concatenate([s.X for s in shared_sites])
    xhat = Z1 @ np.linalg.lstsq(Z1, X, rcond=None)[0]
    K = len(shared_sites)
    row = 0
    Zs, Ws = [], []
    for k, s in enumerate(shared_sites):
        S = np.zeros((s.n, K))
        S[:, k] = 1
        xh = xhat[row:row + s.n]
        row += s.n
        Zs.append(np.column_stack([xh, xh**2, s.C, S]))
        Ws.append(np.column_stack([s.X, s.X**2, s.C, S]))
    theta, cov, _, _ = stacked_2sls(np.vstack(Zs), np.vstack(Ws), np.concatenate([s.Y for s in shared_sites]), 0)
    assert np.allclose(run.result.theta, theta[:3], atol=TOL)
    assert np.allclose(run.result.cov, cov[:3, :3], atol=TOL)
    assert run.result.diagnostics.first_stage["X"]["conditional"] is False


# --------------------------------------------------------------------------- local-first-stage protocol


def test_local_linear_identity_with_pooled_2sls(sites) -> None:
    run = fm.LocalFirstStageFedMR(basis="linear", robust=True).run(sites)
    pooled, pooled_se, _ = pooled_2sls(raw(sites))
    assert abs(run.result["X"] - pooled) < TOL
    assert abs(run.result.se("X") - pooled_se) < TOL
    assert run.rounds == 2


def test_local_quadratic_identity_with_quadratic_2sls(sites) -> None:
    res = fm.LocalFirstStageFedMR(basis="quadratic", robust=False).run(sites).result
    theta, cov = quadratic_2sls(raw(sites))
    assert np.allclose(res.theta, theta, atol=TOL)
    assert np.allclose(res.cov, cov, atol=TOL)


def test_local_robust_cov_equals_stacked_hc0(sites) -> None:
    res = fm.LocalFirstStageFedMR(basis="linear", robust=True).run(sites).result
    Z, W, Y = stack_with_dummies(sites, lambda s: local_first_stage(s)[:, None], lambda s: s.X[:, None])
    theta, cov, hc0, sigma2 = stacked_2sls(Z, W, Y, absorbed=0)
    assert abs(res["X"] - theta[0]) < TOL
    assert abs(res.cov[0, 0] - cov[0, 0]) < TOL
    assert abs(res.robust_cov[0, 0] - hc0[0, 0]) < TOL
    assert abs(res.sigma2 - sigma2) < TOL and abs(res.rss - sigma2 * res.df_resid) < 1e-6


# --------------------------------------------------------------------------- invariances and transport


def test_site_order_invariance(sites) -> None:
    a = fm.LocalFirstStageFedMR(robust=True).run(sites).result
    b = fm.LocalFirstStageFedMR(robust=True).run(sites[::-1]).result
    assert abs(a["X"] - b["X"]) < TOL and abs(a.se("X", True) - b.se("X", True)) < TOL


def test_row_partition_across_transport_clients_is_invariant(sites) -> None:
    """Splitting one logical site's rows across two transport clients, keeping its first
    stage and design columns, does not change the sums or the fit. (Splitting into two
    logical sites would not be invariant: that is a different model.)"""
    d = design_generated(sites[0], local_first_stage(sites[0]), "linear")
    h = d.n // 2
    def part(rows):
        return fm.Design(d.site, d.Z[rows], d.W[rows], d.Y[rows], d.z_names, d.w_names,
                         d.z_roles, d.w_roles, absorbed=0)
    whole = fm.aggregate([fm.site_stats(d)])
    split = fm.aggregate([fm.site_stats(part(slice(0, h))), fm.site_stats(part(slice(h, None)))])
    for k in ("A", "B", "c", "D", "e"):
        assert np.allclose(getattr(whole, k), getattr(split, k), atol=TOL)
    assert abs(whole.f - split.f) < 1e-6 and whole.N == split.N


def test_transport_roundtrip(sites) -> None:
    st = fm.site_stats(design_generated(sites[0], local_first_stage(sites[0]), "linear"))
    back = fm.SiteStats.from_transport(st.arrays(), st.meta())
    assert back.n == st.n and back.z_names == st.z_names and back.z_roles == st.z_roles
    assert np.array_equal(back.A, st.A) and back.f == st.f and back.absorbed == st.absorbed


def test_oracle_column_rejected_by_loader() -> None:
    with pytest.raises(ValueError):
        fm.load_site_csv(FED / CONTINUOUS[0] / "site01.csv", covariates=("U",))


def test_under_identification_raises(shared_sites) -> None:
    s = shared_sites[0]
    d = fm.Design(s.name, np.empty((s.n, 0)), s.X[:, None] - s.X.mean(), s.Y - s.Y.mean(), [], ["X"], [],
                  [fm.Role.ENDOGENOUS])
    with pytest.raises(fm.IdentificationError):
        fm.fit(fm.aggregate([fm.site_stats(d)]))


def test_collinear_instruments_raise(shared_sites) -> None:
    s = shared_sites[0]
    d = design_shared(s)
    Z = np.column_stack([d.Z, d.Z[:, :1]])
    bad = fm.Design(s.name, Z, d.W, d.Y, d.z_names + ["dup"], d.w_names, d.z_roles + [fm.Role.INSTRUMENT], d.w_roles)
    with pytest.raises(fm.IdentificationError):
        fm.fit(fm.aggregate([fm.site_stats(bad)]))


# --------------------------------------------------------------------------- cross-fitting


def test_crossfit_is_generated_instrument_iv_not_regression(sites) -> None:
    """The estimating equation sum xhat_oof (Y - theta X) = 0: theta = xhat_oof'Y / xhat_oof'X
    within site, not the OLS slope of Y on xhat_oof."""
    from flamingo_fedmr.protocols import crossfit_xhat_local, fold_ids
    res = fm.LocalFirstStageFedMR(crossfit=5, robust=False).run(sites).result
    numerator = denominator = 0.0
    for i, s in enumerate(sites):
        xh = crossfit_xhat_local(s, fold_ids(s.n, 5, i), 5)
        xh_c, X_c, Y_c = xh - xh.mean(), s.X - s.X.mean(), s.Y - s.Y.mean()
        numerator += xh_c @ Y_c
        denominator += xh_c @ X_c
    assert abs(res["X"] - numerator / denominator) < TOL


def test_shared_crossfit_runs(shared_sites) -> None:
    run = fm.SharedInstrumentFedMR(crossfit=4, robust=True).run(shared_sites)
    assert run.rounds == 3 and np.isfinite(run.result.se("X", True))
