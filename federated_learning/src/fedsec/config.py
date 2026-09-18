"""Config schema for the robust / private federation. Every section is off by default.

A config is a YAML file with any subset of these sections; missing keys take
the defaults below. `--secure_set section.key=value` on job.py overrides single
keys. `validate()` rejects combinations whose guarantees do not hold (for
example a coordinate-wise median behind secure aggregation, which never sees
the individual updates it would need) instead of silently changing them.

    name: my_scenario
    weights: reported          # reported | registered | uniform: aggregation weight per site
    attack:  {enabled: false, kind: sign_flip, n_malicious: 1, select: largest, ...}
    defense: {enabled: false, rule: median, norm_bound: null, ...}
    dp:      {enabled: false, mode: central, clip_norm: 0.5, noise_multiplier: 1.0, delta: 1.0e-5}
    secagg:  {enabled: false, frac_bits: 24}
    leakage: {enabled: false, mia_max_members: 2000}

The assumption behind each switch is in ROBUST_PRIVATE.md; the ones that change what
an estimate means are repeated next to the field here.
"""

import copy
import dataclasses
from dataclasses import dataclass, field
from typing import List, Optional, Union

import yaml

WEIGHTS = ("reported", "registered", "uniform")
ATTACKS = ("sign_flip", "scale", "gaussian", "outcome_flip", "outcome_shuffle", "none")
MODEL_ATTACKS = ("sign_flip", "scale", "gaussian")          # act on the update the site sends
DATA_ATTACKS = ("outcome_flip", "outcome_shuffle")          # act on the site's training data
RULES = ("fedavg", "median", "trimmed_mean", "krum", "multi_krum", "geometric_median")
DP_MODES = ("central", "local", "distributed")


@dataclass
class AttackConfig:
    enabled: bool = False
    # sign_flip   send -scale * update                  (model poisoning)
    # scale       send  scale * update                  (model replacement / boosting)
    # gaussian    send  N(0, noise_std^2) per coordinate (random update)
    # outcome_flip     train on the negated model output: Y -> -Y (continuous), Y -> 1 - Y (binary),
    #                  reversed hazard ordering (survival); the update is then sent honestly
    # outcome_shuffle  train on outcomes permuted within the site: breaks X -> outcome, pulls f toward flat
    # none             malicious sites behave honestly except for weight_multiplier
    kind: str = "sign_flip"
    sites: List[str] = field(default_factory=list)   # explicit malicious sites; overrides n_malicious/select
    n_malicious: int = 1
    select: str = "largest"          # largest | smallest | random (by training-set size)
    scale: float = 1.0               # sign_flip and scale
    noise_std: float = 1.0           # gaussian
    weight_multiplier: float = 1.0   # reported sample size is multiplied by this (weight inflation)
    start_round: int = 0             # honest before this round
    # A malicious site controls its own software. False: it skips client-side clipping and
    # client-side DP noise. True: it runs them on its poisoned update like everyone else.
    follows_protocol: bool = False


@dataclass
class DefenseConfig:
    enabled: bool = False
    rule: str = "median"             # see RULES; fedavg = weighted mean
    trim_fraction: float = 0.1       # trimmed_mean: fraction of sites cut at each end, per coordinate
    krum_f: int = 1                  # krum / multi_krum: number of Byzantine sites assumed
    multi_krum_m: Optional[int] = None   # sites kept by multi_krum; default K - krum_f
    gm_max_iter: int = 100           # geometric_median (smoothed Weiszfeld)
    gm_tol: float = 1e-7
    gm_nu: float = 1e-6              # smoothing: distances below this count as this
    # Server-side update-norm bounding before the rule: null (off), a number (absolute L2 bound),
    # or "adaptive" (norm_bound_multiplier x the median L2 norm of this round's updates).
    norm_bound: Union[None, float, str] = None
    norm_bound_multiplier: float = 1.0


@dataclass
class DPConfig:
    enabled: bool = False
    # Site-level (client-level) differential privacy of the update stream. Each training round
    # every site's update is L2-clipped to clip_norm and Gaussian noise is added:
    # central      server clips every received update and adds N(0, (z * C * max_w)^2) to the
    #              weighted sum. Protects the released global model; the server itself sees raw updates.
    # local        each site clips and adds N(0, (z * C)^2) to its own update before sending.
    #              Protects against the server and anyone on the wire; much more total noise.
    # distributed  each site clips and adds its 1/min_honest share of the central noise. With
    #              secagg on, the server only ever sees the noisy sum (same epsilon as central
    #              against the server too); without secagg each message is only weakly protected.
    mode: str = "central"
    # Chosen from a non-private pilot run on the simulated data (honest update norms 0.3-2.3, median
    # ~0.8 over rounds). With real data C must come from public or simulated data, since choosing it
    # from the private updates leaks; adaptive clipping with a private quantile is not implemented.
    clip_norm: float = 0.5
    noise_multiplier: float = 1.0    # z: noise std in units of the per-site sensitivity
    delta: float = 1e-5
    # distributed: sites split the noise as if at least this many honest sites add their share.
    # null = all K sites. If fewer honest sites actually add noise, the accountant reports the
    # smaller effective noise.
    min_honest: Optional[int] = None


@dataclass
class SecAggConfig:
    enabled: bool = False
    # Fixed-point encoding: x -> round(x * 2^frac_bits) as an integer mod 2^64. Quantization
    # error per coordinate is at most 2^-(frac_bits+1) before weighting.
    frac_bits: int = 24


@dataclass
class LeakageConfig:
    enabled: bool = False
    mia_max_members: int = 2000      # training records scored by the membership-inference probe


@dataclass
class SecureConfig:
    name: str = "baseline"
    description: str = ""
    # reported    FedAvg's default: the sample size each site reports (a malicious site can inflate it)
    # registered  sample sizes the server holds from registration (the public training-split sizes)
    # uniform     every site weighs 1/K
    weights: str = "reported"
    attack: AttackConfig = field(default_factory=AttackConfig)
    defense: DefenseConfig = field(default_factory=DefenseConfig)
    dp: DPConfig = field(default_factory=DPConfig)
    secagg: SecAggConfig = field(default_factory=SecAggConfig)
    leakage: LeakageConfig = field(default_factory=LeakageConfig)

    @property
    def active(self) -> bool:
        """False when every component is off: the engines then run their original code path."""
        return (self.weights != "reported" or self.attack.enabled or self.defense.enabled or self.dp.enabled
                or self.secagg.enabled or self.leakage.enabled)

    @property
    def rule(self) -> str:
        return self.defense.rule if self.defense.enabled else "fedavg"

    @property
    def norm_bound(self):
        return self.defense.norm_bound if self.defense.enabled else None

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


_SECTIONS = {"attack": AttackConfig, "defense": DefenseConfig, "dp": DPConfig, "secagg": SecAggConfig,
             "leakage": LeakageConfig}


def from_dict(d: Optional[dict]) -> SecureConfig:
    d = copy.deepcopy(d or {})
    kwargs = {}
    for key, value in d.items():
        if key in _SECTIONS:
            cls = _SECTIONS[key]
            known = {f.name for f in dataclasses.fields(cls)}
            unknown = set(value or {}) - known
            if unknown:
                raise ValueError(f"unknown key(s) in '{key}': {sorted(unknown)}; known: {sorted(known)}")
            kwargs[key] = cls(**(value or {}))
        elif key in ("name", "description", "weights"):
            kwargs[key] = value
        else:
            raise ValueError(f"unknown top-level key '{key}'")
    return SecureConfig(**kwargs)


def apply_overrides(d: dict, overrides: List[str]) -> dict:
    """`section.key=value` strings onto a config dict; values are parsed as YAML scalars/lists."""
    d = copy.deepcopy(d or {})
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override '{item}' is not key=value")
        path, raw = item.split("=", 1)
        keys = path.strip().split(".")
        node = d
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = yaml.safe_load(raw)
    return d


def load_config(path: Optional[str] = None, overrides: Optional[List[str]] = None) -> SecureConfig:
    d = {}
    if path:
        with open(path) as f:
            d = yaml.safe_load(f) or {}
    return from_dict(apply_overrides(d, overrides))


def malicious_sites(cfg: SecureConfig, sizes: dict, seed: int = 0) -> List[str]:
    """Which sites are malicious. sizes: {site: training-set size}."""
    a = cfg.attack
    if not a.enabled:
        return []
    if a.sites:
        missing = set(a.sites) - set(sizes)
        if missing:
            raise ValueError(f"attack.sites not in this federation: {sorted(missing)}")
        return sorted(a.sites)
    names = sorted(sizes)
    if a.select == "largest":
        order = sorted(names, key=lambda s: (-sizes[s], s))
    elif a.select == "smallest":
        order = sorted(names, key=lambda s: (sizes[s], s))
    elif a.select == "random":
        import numpy as np
        order = list(np.random.default_rng([seed, 5]).permutation(names))
    else:
        raise ValueError(f"attack.select must be largest | smallest | random, got {a.select!r}")
    return sorted(order[:a.n_malicious])


def validate(cfg: SecureConfig, sites: List[str], warn=print) -> None:
    """Raise on combinations whose guarantees do not hold; warn when an aggregator's
    robustness assumption (enough honest sites) is violated by the configured attack."""
    K = len(sites)
    if cfg.weights not in WEIGHTS:
        raise ValueError(f"weights must be one of {WEIGHTS}, got {cfg.weights!r}")
    a, r, dp, sa = cfg.attack, cfg.defense, cfg.dp, cfg.secagg
    if a.enabled:
        if a.kind not in ATTACKS:
            raise ValueError(f"attack.kind must be one of {ATTACKS}, got {a.kind!r}")
        n_mal = len(a.sites) if a.sites else a.n_malicious
        if not 1 <= n_mal < K:
            raise ValueError(f"need 1 <= malicious sites < {K}, got {n_mal}")
        if a.weight_multiplier <= 0:
            raise ValueError("attack.weight_multiplier must be > 0")
    else:
        n_mal = 0
    if r.enabled:
        if r.rule not in RULES:
            raise ValueError(f"defense.rule must be one of {RULES}, got {r.rule!r}")
        if not 0 <= r.trim_fraction < 0.5:
            raise ValueError("defense.trim_fraction must be in [0, 0.5)")
        if r.rule == "trimmed_mean" and int(r.trim_fraction * K) == 0:
            warn(f"[fedsec] warning: trim_fraction {r.trim_fraction} trims 0 of {K} sites per end; "
                 "trimmed_mean equals the weighted mean")
        if r.norm_bound is not None and r.norm_bound != "adaptive":
            if not isinstance(r.norm_bound, (int, float)) or r.norm_bound <= 0:
                raise ValueError("defense.norm_bound must be null, 'adaptive', or a positive number")
        if r.rule in ("krum", "multi_krum"):
            if K < 2 * r.krum_f + 3:
                warn(f"[fedsec] warning: Krum's guarantee needs K >= 2f + 3; K={K}, f={r.krum_f}")
            if r.multi_krum_m is not None and not 1 <= r.multi_krum_m <= K:
                raise ValueError("defense.multi_krum_m must be in [1, K]")
            if n_mal > r.krum_f:
                warn(f"[fedsec] warning: {n_mal} malicious sites but Krum assumes at most f={r.krum_f}")
        if r.rule in ("median", "geometric_median") and n_mal * 2 >= K:
            warn(f"[fedsec] warning: {r.rule} needs an honest majority; {n_mal} of {K} sites are malicious")
        if r.rule == "trimmed_mean" and n_mal > int(r.trim_fraction * K):
            warn(f"[fedsec] warning: trimmed_mean cuts {int(r.trim_fraction * K)} per end but "
                 f"{n_mal} sites are malicious")
    if dp.enabled:
        if dp.mode not in DP_MODES:
            raise ValueError(f"dp.mode must be one of {DP_MODES}, got {dp.mode!r}")
        if dp.clip_norm <= 0 or dp.noise_multiplier <= 0 or not 0 < dp.delta < 1:
            raise ValueError("dp needs clip_norm > 0, noise_multiplier > 0, 0 < delta < 1")
        if cfg.weights == "reported":
            # the noise is calibrated to the largest weight; self-reported weights are not bounded
            raise ValueError("dp needs weights: registered or uniform (the sensitivity of the weighted sum "
                             "depends on the largest weight, so the weights must be fixed in advance)")
        if dp.mode in ("central", "distributed") and cfg.rule != "fedavg":
            raise ValueError(f"dp.mode={dp.mode} is accounted for the clipped weighted mean only; "
                             f"defense.rule={cfg.rule} has a different sensitivity. Use dp.mode=local "
                             "(robust rules are then post-processing) or defense.rule=fedavg")
        if dp.mode in ("central", "distributed") and cfg.norm_bound is not None:
            raise ValueError("dp.clip_norm already bounds every update; set defense.norm_bound: null")
        if dp.min_honest is not None and not 1 <= dp.min_honest <= K:
            raise ValueError("dp.min_honest must be in [1, K]")
        if dp.mode == "distributed" and not sa.enabled:
            warn("[fedsec] warning: dp.mode=distributed without secagg: the server sees each site's "
                 "message with only 1/min_honest of the noise; see epsilon_server in privacy.json")
    if sa.enabled:
        if cfg.rule != "fedavg":
            raise ValueError(f"secagg reveals only the sum of updates; defense.rule={cfg.rule} needs the "
                             "individual updates. Use defense.rule=fedavg (or disable secagg)")
        if cfg.norm_bound is not None:
            raise ValueError("secagg hides individual updates, so the server cannot bound their norms; "
                             "use dp (clipping then runs at each site) or disable secagg")
        if not 8 <= sa.frac_bits <= 40:
            raise ValueError("secagg.frac_bits must be in [8, 40]")
