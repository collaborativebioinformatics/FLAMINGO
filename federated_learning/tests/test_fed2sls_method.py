"""Checks for the `fed2sls` method's engine (src/fed2sls_engine.py): the local engine equals the
package run in-process and the pooled fit, and the files it writes have the layout plots.py
and job.py read. The NVFlare transport itself is exercised by `job.py --method fed2sls`, which
asserts the same identities after every simulator run.

    uv run --with pytest pytest tests/test_fed2sls_method.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

import flamingo_fedmr as fm  # noqa: E402
import fed2sls_engine as E  # noqa: E402
from tasks import X_GRID  # noqa: E402

FED_DIR = os.path.join(os.path.dirname(os.path.dirname(HERE)), "data", "simulated_data", "federated")
LINEAR = os.path.join(FED_DIR, "linear")
SHARED = os.path.join(FED_DIR, "linear_shared")

pytestmark = pytest.mark.skipif(not os.path.isdir(LINEAR), reason="simulated federated data not generated")


def _no_simulator(*args, **kwargs):
    raise AssertionError("the local engine must not start the simulator")


@pytest.fixture(scope="module")
def local_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("results")
    row = E.run("linear", LINEAR, str(out), str(out / "ws"), "local", "linear", 0, 0, _no_simulator, 0.05)
    return out, row


def test_local_engine_equals_package_and_pooled(local_run):
    out, row = local_run
    est = json.load(open(out / "estimates.linear.json"))
    ref = fm.LocalFirstStageFedMR(basis="linear", robust=True).run(fm.load_sites(LINEAR)).result
    assert est["w_names"] == ref.w_names
    assert np.allclose(est["theta"], ref.theta, atol=0, rtol=0)
    assert abs(row["theta_X"] - ref["X"]) == 0.0
    assert abs(row["se_X"] - ref.se("X")) < 1e-15
    assert abs(row["robust_se_X"] - ref.se("X", True)) < 1e-15
    assert row["max_diff_vs_pooled"] < E.TOL
    assert "max_diff_vs_inprocess" not in row       # nothing to compare against without the transport


def test_files_have_the_layout_plots_read(local_run):
    out, row = local_run
    curves = pd.read_csv(out / "curves.csv")
    metrics = pd.read_csv(out / "metrics.csv")
    assert list(curves.columns) == E.CURVE_COLUMNS
    assert np.allclose(curves.x, X_GRID)
    assert (curves["round"] == row["round"]).all() and (curves.site == "all").all()
    assert len(metrics) == 1
    for col in ("site", "round", "stage", "n_train", "n_test", "theta_X", "se_X", "robust_se_X", "first_stage_F",
                "protocol", "basis", "crossfit"):
        assert col in metrics.columns, col
    assert metrics.protocol.iloc[0] == "local" and metrics.basis.iloc[0] == "linear"


def test_curve_is_the_slope_through_the_origin_with_an_exact_linear_band(local_run):
    out, row = local_run
    curves = pd.read_csv(out / "curves.csv")
    x = curves.x.to_numpy()
    assert np.allclose(curves.f, row["theta_X"] * x)
    half = 1.96 * row["robust_se_X"] * np.abs(x)
    assert np.allclose(curves.f_hi - curves.f, half) and np.allclose(curves.f - curves.f_lo, half)
    assert curves.loc[np.isclose(x, 0.0), "f"].abs().max() < 1e-12


def test_quadratic_basis_curve_uses_both_coefficients():
    est = {"w_names": ["X", "X2"], "theta": [0.3, -0.05], "se": [0.01, 0.02], "robust_se": None}
    c = E.curve_table(est, rounds=1)
    x = c.x.to_numpy()
    assert np.allclose(c.f, 0.3 * x - 0.05 * x**2)
    assert np.allclose(c.f_hi - c.f, 1.96 * np.sqrt((x * 0.01) ** 2 + (x**2 * 0.02) ** 2))


def test_shared_protocol_is_linear_only_and_others_are_open():
    shared = {"shared_snps": True}
    assert E.supported(shared, "linear", 0) is None
    assert E.supported(shared, "quadratic", 0)
    assert E.supported(shared, "linear", 5)
    assert E.supported({}, "quadratic", 5) is None
    assert E.protocol_name(shared) == "shared" and E.protocol_name({}) == "local"


@pytest.mark.skipif(not os.path.isdir(SHARED), reason="linear_shared set not generated")
def test_shared_set_runs_locally_without_a_pooled_gap(tmp_path):
    row = E.run("linear_shared", SHARED, str(tmp_path), str(tmp_path / "ws"), "local", "linear", 0, 0,
                _no_simulator, 0.05)
    assert row["protocol"] == "shared"
    assert "max_diff_vs_pooled" not in row
    ref = fm.SharedInstrumentFedMR(basis="linear", robust=True).run(fm.load_sites(SHARED)).result
    assert abs(row["theta_X"] - ref["X"]) == 0.0


def test_spec_tag():
    assert E.spec_tag("linear", 0) == "linear"
    assert E.spec_tag("quadratic", 5) == "quadratic.cf5"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
