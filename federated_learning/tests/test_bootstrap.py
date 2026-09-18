"""Checks for the bootstrap band arithmetic (src/bootstrap.py). Run with either

    uv run python tests/test_bootstrap.py
    uv run pytest federated_learning/tests/test_bootstrap.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import bootstrap  # noqa: E402
from plots import last_band  # noqa: E402
from tasks import X_GRID  # noqa: E402


def _boot(n_rep=400, seed=0):
    rng = np.random.default_rng(seed)
    x = X_GRID.astype(float)
    slopes = rng.normal(0.3, 0.05, n_rep)
    return pd.DataFrame([(b, xv, s * xv) for b, s in enumerate(slopes) for xv in x], columns=["rep", "x", "f"])


def test_band_is_pointwise_percentile():
    boot = _boot()
    bnd = bootstrap.band(boot, 0.9).set_index("x")
    at1 = boot[np.isclose(boot.x, 1.0)].f
    assert np.isclose(bnd.lo[bnd.index[np.isclose(bnd.index, 1.0)][0]], at1.quantile(0.05))
    assert np.isclose(bnd.hi[bnd.index[np.isclose(bnd.index, 1.0)][0]], at1.quantile(0.95))
    # anchored curves: zero width at X = 0, widening with |X|
    zero = bnd.loc[bnd.index[np.isclose(bnd.index, 0.0)][0]]
    assert abs(zero.hi - zero.lo) < 1e-12
    assert (bnd.hi - bnd.lo).iloc[-1] > (bnd.hi - bnd.lo).iloc[len(bnd) // 2 + 5]


def test_attach_band_round_trips_through_last_band():
    """plots.last_band re-anchors at the point estimate's f(0); attach_band must pre-shift so the
    drawn band equals the anchored bootstrap band, whatever the point curve's level."""
    x = X_GRID.astype(float)
    rows = [(s, r, xv, 5.0 + 0.3 * xv + 0.01 * r) for s in ("site01", "site02") for r in range(3) for xv in x]
    curves = pd.DataFrame(rows, columns=["site", "round", "x", "f"])
    bnd = bootstrap.band(_boot(), 0.95)
    out = bootstrap.attach_band(curves, bnd)
    assert out.loc[out["round"] < 2, "f_lo"].isna().all()           # only the last round carries the band
    bx, lo, hi = last_band(out)
    assert np.allclose(lo, bnd.sort_values("x").lo.to_numpy(), atol=1e-9)
    assert np.allclose(hi, bnd.sort_values("x").hi.to_numpy(), atol=1e-9)


if __name__ == "__main__":
    tests = [(k, v) for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    print(f"{len(tests)} tests passed")
