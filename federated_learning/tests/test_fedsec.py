"""Checks for the fedsec building blocks. Run with either

    uv run python tests/test_fedsec.py
    uv run --with pytest pytest tests/
"""

import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from fedsec import aggregators, dp, secagg  # noqa: E402
from fedsec.config import from_dict, validate, malicious_sites  # noqa: E402
from fedsec.protocol import ParamSpec, Message, ServerProtocol, ClientProtocol  # noqa: E402

SITES = [f"site{i:02d}" for i in range(1, 11)]
SIZES = {s: 1000 + 500 * i for i, s in enumerate(SITES)}


def _updates(K=10, P=50, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(0.3, 0.05, size=(K, P)), np.full(K, 1.0 / K)


def test_fedavg_is_weighted_mean():
    U, _ = _updates()
    w = np.arange(1, 11, dtype=float)
    agg, _ = aggregators.aggregate(U, w, "fedavg")
    assert np.allclose(agg, (w / w.sum()) @ U)


def test_robust_rules_resist_one_outlier():
    U, w = _updates()
    honest = U[1:].mean(axis=0)
    U[0] = -100.0                                            # one site sends a huge flipped update
    fed, _ = aggregators.aggregate(U, w, "fedavg")
    assert np.abs(fed - honest).max() > 5
    for rule in ("median", "trimmed_mean", "krum", "multi_krum", "geometric_median"):
        agg, info = aggregators.aggregate(U, w, rule, trim_fraction=0.1, krum_f=1)
        assert np.abs(agg - honest).max() < 0.1, rule
        assert info["flagged"][0], rule
        if rule != "krum":                                   # krum keeps one update and drops the rest
            assert info["flagged"][1:].mean() < 0.5, rule


def test_weighted_median_and_norm_bound():
    U = np.array([[0.0], [1.0], [2.0], [10.0]])
    agg, _ = aggregators.aggregate(U, np.array([0.1, 0.1, 0.7, 0.1]), "median")
    assert agg[0] == 2.0
    agg, info = aggregators.aggregate(U, np.full(4, 0.25), "fedavg", norm_bound_value=2.0)
    assert np.isclose(agg[0], (0 + 1 + 2 + 2) / 4) and info["clipped"].tolist() == [False, False, False, True]


def test_secagg_sum_exact_and_messages_uninformative():
    rng = np.random.default_rng(1)
    K, P, bits = 6, 200, 24
    xs = [rng.normal(size=P) for _ in range(K)]
    key = secagg.session_key(3)
    msgs = [secagg.mask_message(x, i, K, 0, key, bits) for i, x in enumerate(xs)]
    total = secagg.unmask_sum(msgs, bits)
    assert np.abs(total - sum(xs)).max() < K * 2.0 ** -bits
    # a single masked message decodes to noise unrelated to the site's vector
    seen = secagg.decode(msgs[0], bits)
    assert abs(np.corrcoef(seen, xs[0])[0, 1]) < 0.3
    # a missing site leaves masks uncancelled
    assert np.abs(secagg.unmask_sum(msgs[1:], bits) - sum(xs[1:])).max() > 1e3


def test_gdp_accountant():
    # delta(eps) is decreasing; eps(mu, delta) inverts it
    mu = 1.0
    eps = dp.gdp_epsilon(mu, 1e-5)
    assert abs(dp.gdp_delta(eps, mu) - 1e-5) < 1e-8
    assert dp.gdp_epsilon(0.5, 1e-5) < eps < dp.gdp_epsilon(2.0, 1e-5)
    assert dp.gdp_epsilon(math.inf, 1e-5) == math.inf
    # the GDP bound is tighter than the RDP bound for the same Gaussian composition
    rdp = min(a * mu**2 / 2 + math.log(1 / 1e-5) / (a - 1) for a in np.linspace(1.01, 200, 20000))
    assert eps <= rdp + 1e-9


def test_accounting_modes():
    w = {s: 0.1 for s in SITES}
    base = dict(enabled=True, clip_norm=0.1, noise_multiplier=2.0, delta=1e-5)
    c = from_dict({"weights": "uniform", "dp": {**base, "mode": "central"}})
    rep = dp.account(c.dp, False, 5, w, SITES)
    assert math.isfinite(rep["epsilon_aggregate"]) and rep["epsilon_server"] == math.inf
    d = from_dict({"weights": "uniform", "dp": {**base, "mode": "distributed"}})
    with_sa = dp.account(d.dp, True, 5, w, SITES)
    assert abs(with_sa["epsilon_server"] - rep["epsilon_aggregate"]) < 1e-9
    without_sa = dp.account(d.dp, False, 5, w, SITES)
    assert without_sa["epsilon_server"] > with_sa["epsilon_server"]
    fewer = dp.account(d.dp, True, 5, w, SITES[:7])            # 3 sites skip their noise share
    assert fewer["epsilon_aggregate"] > with_sa["epsilon_aggregate"]


def test_validation_rejects_broken_guarantees():
    bad = [
        {"secagg": {"enabled": True}, "defense": {"enabled": True, "rule": "median"}},
        {"secagg": {"enabled": True}, "defense": {"enabled": True, "rule": "fedavg", "norm_bound": 1.0}},
        {"dp": {"enabled": True}},                                            # weights: reported
        {"weights": "registered", "dp": {"enabled": True, "mode": "central"},
         "defense": {"enabled": True, "rule": "krum"}},
        {"attack": {"enabled": True, "n_malicious": 10}},
    ]
    for d in bad:
        try:
            validate(from_dict(d), SITES, warn=lambda *_: None)
        except ValueError:
            continue
        raise AssertionError(f"accepted {d}")
    validate(from_dict({"weights": "registered", "dp": {"enabled": True, "mode": "local"},
                        "defense": {"enabled": True, "rule": "median"}}), SITES, warn=lambda *_: None)
    try:
        from_dict({"attack": {"enabled": True, "typo": 1}})
    except ValueError:
        pass
    else:
        raise AssertionError("unknown key accepted")


def test_malicious_selection():
    cfg = from_dict({"attack": {"enabled": True, "n_malicious": 2, "select": "largest"}})
    assert malicious_sites(cfg, SIZES) == ["site09", "site10"]
    cfg = from_dict({"attack": {"enabled": True, "sites": ["site03"]}})
    assert malicious_sites(cfg, SIZES) == ["site03"]
    assert malicious_sites(from_dict({}), SIZES) == []


def test_paramspec_roundtrip_and_flare_message():
    state = {"a.weight": torch.randn(3, 2), "a.bias": torch.randn(3)}
    spec = ParamSpec(state)
    v = spec.flatten(state)
    back = spec.unflatten(v)
    assert all(torch.equal(back[k], state[k]) for k in state)
    assert ParamSpec(spec=spec.to_list()).items == spec.items
    m = Message("masked", np.array([1, 2**63 + 5, 2**64 - 1], dtype=np.uint64), 1.0, 2)
    m2 = Message.from_flare(*m.to_flare())
    assert m2.kind == "masked" and np.array_equal(m2.vec, m.vec)


class _FakeSite:
    def __init__(self, n):
        self.n_train = n


def test_secagg_protocol_equals_plain_fedavg():
    """End to end through ClientProtocol/ServerProtocol: secagg returns the same aggregate as plain FedAvg."""
    state = {"w": torch.zeros(20)}
    spec = ParamSpec(state)
    rng = np.random.default_rng(0)
    locals_ = {s: {"w": torch.tensor(rng.normal(size=20), dtype=torch.float32)} for s in SITES}
    out = {}
    for on in (False, True):
        cfg = from_dict({"secagg": {"enabled": on}, "leakage": {"enabled": False}})
        clients = {s: ClientProtocol(cfg, s, SITES, SIZES, 0, spec) for s in SITES}
        msgs = {s: clients[s].make_message(0, state, locals_[s], _FakeSite(SIZES[s]))[0] for s in SITES}
        out[on], _ = ServerProtocol(cfg, SITES, SIZES, 0, spec).aggregate(0, msgs)
    assert np.abs(out[True] - out[False]).max() < 1e-6


if __name__ == "__main__":
    tests = [(k, v) for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    print(f"{len(tests)} tests passed")
