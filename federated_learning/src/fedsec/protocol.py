"""Client and server message pipelines, shared by the local engine and NVFlare.

A site's round, after fedsite.Site.run_round has trained the local model:

    u = local - global                                  the true update
    v = attack(u)             malicious sites only, model-poisoning attacks
    v = clip(v, C)            dp at the site (local / distributed mode, or any dp mode behind secagg)
    v = v + N(0, s_k^2)       dp noise at the site (local / distributed)
    message = v, weight       plain; or
    message = mask(enc([r v, r]))    secagg: pre-weighted, fixed-point, masked

The server, per round:

    plain   U = stack(v_k); clip rows to C (central dp); norm bound; robust rule -> aggregate update
    masked  sum of messages -> sum_k r_k v_k / sum_k r_k
    then    + N(0, s^2)                                 central dp
            global <- global + aggregate update

With the rule fedavg and no attack, dp, or secagg this is FedAvg on updates:
global + sum_k w_k (local_k - global) = sum_k w_k local_k.

Weights. reported: the n each site reports (FedAvg's NUM_STEPS_CURRENT_ROUND).
registered: the training-split sizes the server holds. uniform: 1/K. Behind
secagg the server cannot check any weight, so a malicious site's inflated
weight gets through in every mode.
"""

import copy
import json
import math
import os
from dataclasses import dataclass

import numpy as np
import torch

from . import aggregators, dp as dpm, secagg
from .attacks import DataPoisoning, poison_update
from .config import DATA_ATTACKS, MODEL_ATTACKS, malicious_sites, validate
from .leakage import cosine, mia_auc

CLIENT_LOG = "clients.csv"
SERVER_LOG = "server.csv"
PRIVACY_FILE = "privacy.json"


class ParamSpec:
    """Flattens a state dict to one float64 vector and back, in a fixed key order."""

    def __init__(self, state=None, spec=None):
        if spec is not None:
            self.items = [(k, tuple(s), d) for k, s, d in spec]
        else:
            self.items = [(k, tuple(v.shape), str(v.dtype).replace("torch.", "")) for k, v in state.items()]
        self.size = int(sum(int(np.prod(s)) for _, s, _ in self.items))

    def to_list(self):
        return [[k, list(s), d] for k, s, d in self.items]

    def flatten(self, state):
        return np.concatenate([np.asarray(state[k].detach().cpu().numpy() if torch.is_tensor(state[k])
                                          else state[k], dtype=np.float64).ravel() for k, _, _ in self.items])

    def unflatten(self, vec):
        out, i = {}, 0
        for k, s, d in self.items:
            n = int(np.prod(s))
            out[k] = torch.tensor(vec[i:i + n].reshape(s), dtype=getattr(torch, d))
            i += n
        return out

    def unflatten_numpy(self, vec):
        """As unflatten, but numpy arrays (NVFlare's server holds the global model as numpy)."""
        return {k: v.numpy() for k, v in self.unflatten(vec).items()}

    def add(self, state, vec):
        return self.unflatten(self.flatten(state) + vec)


@dataclass
class Message:
    kind: str                 # plain | masked | eval
    vec: np.ndarray           # float64 update (plain), uint64 masked integers (masked)
    weight: float = 1.0       # reported weight (plain); inside vec for masked
    round: int = 0

    def to_flare(self):
        vec = self.vec.view(np.int64) if self.kind == "masked" else self.vec
        return ({"fedsec_vec": torch.from_numpy(np.ascontiguousarray(vec))},
                {"fedsec_kind": self.kind, "fedsec_weight": float(self.weight), "fedsec_round": int(self.round)})

    @classmethod
    def from_flare(cls, params, meta):
        vec = params["fedsec_vec"]
        vec = vec.detach().cpu().numpy() if torch.is_tensor(vec) else np.asarray(vec)
        kind = meta["fedsec_kind"]
        vec = vec.astype(np.int64).view(np.uint64) if kind == "masked" else vec.astype(np.float64)
        return cls(kind, vec, float(meta["fedsec_weight"]), int(meta["fedsec_round"]))


def public_weights(cfg, sizes):
    """Normalized weights the server can fix in advance: registered sizes, or uniform."""
    sites = sorted(sizes)
    if cfg.weights == "uniform":
        return {s: 1.0 / len(sites) for s in sites}
    total = float(sum(sizes.values()))
    return {s: sizes[s] / total for s in sites}


def noisy_sites(cfg, sites, bad):
    """Sites that add their client-side dp noise: honest ones, plus malicious ones that follow the protocol."""
    return [s for s in sites if s not in bad or cfg.attack.follows_protocol]


def privacy_report(cfg, sites, sizes, rounds, seed):
    bad = malicious_sites(cfg, sizes, seed)
    rep = dpm.account(cfg.dp, cfg.secagg.enabled, rounds, public_weights(cfg, sizes), noisy_sites(cfg, sites, bad))
    rep.update({"config": cfg.to_dict(), "malicious_sites": bad, "seed": seed})
    return rep


def write_privacy(path, report):
    """privacy.json; an infinite epsilon (no protection) is written as the string "inf"."""
    clean = {k: (str(v) if isinstance(v, float) and not math.isfinite(v) else v) for k, v in report.items()}
    with open(path, "w") as f:
        json.dump(clean, f, indent=1)


class ClientProtocol:
    def __init__(self, cfg, site_name, sites, sizes, seed, spec):
        self.cfg, self.name, self.sites, self.spec, self.seed = cfg, site_name, list(sites), spec, seed
        self.index = self.sites.index(site_name)
        self.K = len(self.sites)
        self.sizes = sizes
        self.malicious = site_name in malicious_sites(cfg, sizes, seed)
        self.pub_w = public_weights(cfg, sizes)
        self.plan = dpm.noise_plan(cfg.dp, cfg.secagg.enabled, self.pub_w) if cfg.dp.enabled else None
        self.noise_rng = np.random.default_rng([seed, 2, self.index])
        self.attack_rng = np.random.default_rng([seed, 1, self.index])
        self.leak_rng = np.random.default_rng([seed, 7, self.index])
        self.key = secagg.session_key(seed) if cfg.secagg.enabled else None
        self.poisoner = None

    def attach(self, site):
        """Called once with the Site; sets up data poisoning for a malicious site."""
        if self.malicious and self.cfg.attack.kind in DATA_ATTACKS:
            self.poisoner = DataPoisoning(site, self.cfg.attack, self.seed, self.index)

    def before_round(self, site, rnd):
        if self.poisoner is not None:
            self.poisoner.prepare(site, rnd)

    def attacking(self, rnd):
        return self.malicious and rnd >= self.cfg.attack.start_round

    def _weight(self, site, attacking):
        """(weight reported in a plain message, pre-weight used behind secagg)."""
        mult = self.cfg.attack.weight_multiplier if attacking else 1.0
        reported = site.n_train * mult
        pre = {"reported": reported, "registered": self.sizes[self.name] * mult, "uniform": 1.0 * mult}
        return reported, pre[self.cfg.weights]

    def make_message(self, rnd, global_state, local_state, site):
        cfg = self.cfg
        u = self.spec.flatten(local_state) - self.spec.flatten(global_state)
        attacking = self.attacking(rnd)
        v = poison_update(u, cfg.attack, self.attack_rng) if attacking and cfg.attack.kind in MODEL_ATTACKS else u
        follows = (not attacking) or cfg.attack.follows_protocol
        clipped = False
        if cfg.dp.enabled and follows:
            if self.plan["clip_at_client"]:
                v, n = dpm.clip(v, cfg.dp.clip_norm)
                clipped = n > cfg.dp.clip_norm
            std = self.plan["client_std"][self.name]
            if std > 0:
                v = v + self.noise_rng.normal(0.0, std, size=v.shape)
        reported, pre = self._weight(site, attacking)
        if cfg.secagg.enabled:
            payload = np.concatenate([pre * v, [pre]])
            msg = Message("masked", secagg.mask_message(payload, self.index, self.K, rnd, self.key,
                                                        cfg.secagg.frac_bits), 1.0, rnd)
        else:
            msg = Message("plain", v, reported, rnd)

        row = {"site": self.name, "round": rnd, "malicious": int(self.malicious), "attacking": int(attacking),
               "update_norm": float(np.linalg.norm(u)), "sent_norm": float(np.linalg.norm(v)),
               "client_clipped": int(clipped), "update_cos": float("nan"), "mia_auc": float("nan")}
        if cfg.leakage.enabled:
            if msg.kind == "masked":
                # the observer's naive reading of a masked message: decode it as if it were the payload
                seen = secagg.decode(msg.vec, cfg.secagg.frac_bits)[:-1]
                row["update_cos"] = cosine(seen, u)
            else:
                row["update_cos"] = cosine(msg.vec, u)
                observed = self.spec.add(global_state, msg.vec)
                from fedsite import make_model
                row["mia_auc"] = mia_auc(site, make_model, global_state, observed,
                                         cfg.leakage.mia_max_members, self.leak_rng)
        return msg, row

    def eval_message(self, rnd):
        """Placeholder reply in the evaluation-only pass (NVFlare needs non-empty params)."""
        return Message("eval", np.zeros(self.spec.size), 0.0, rnd)


class ServerProtocol:
    """Never receives the secagg session key or the sites' data; knows only the public sizes.
    `malicious` is the ground truth, used for logging the detection metrics only."""

    def __init__(self, cfg, sites, sizes, seed, spec):
        self.cfg, self.sites, self.spec = cfg, list(sites), spec
        self.pub_w = public_weights(cfg, sizes)
        self.plan = dpm.noise_plan(cfg.dp, cfg.secagg.enabled, self.pub_w) if cfg.dp.enabled else None
        self.rng = np.random.default_rng([seed, 3])
        self.malicious = set(malicious_sites(cfg, sizes, seed))

    def aggregate(self, rnd, msgs):
        """msgs: {site: Message}. Returns (aggregate update as a flat vector, per-site log rows)."""
        cfg = self.cfg
        present = [s for s in self.sites if s in msgs]
        if all(msgs[s].kind == "eval" for s in present):
            return np.zeros(self.spec.size), []
        if len(present) != len(self.sites):
            raise RuntimeError(f"round {rnd}: {len(present)} of {len(self.sites)} sites reported; "
                               "the protocol assumes every site reports every round")
        rows = []
        if cfg.secagg.enabled:
            total = secagg.unmask_sum([msgs[s].vec for s in present], cfg.secagg.frac_bits)
            agg = total[:-1] / total[-1]
            for s in present:
                rows.append({"round": rnd, "site": s, "malicious": int(s in self.malicious), "rule": "secagg_sum",
                             "weight": float("nan"), "norm": float("nan"), "clipped": 0, "kept": 1.0,
                             "share": float("nan"), "flagged": 0})
        else:
            U = np.stack([msgs[s].vec for s in present])
            if cfg.weights == "reported":
                w = np.array([msgs[s].weight for s in present], dtype=np.float64)
            else:
                w = np.array([self.pub_w[s] for s in present])
            w = w / w.sum()
            server_clipped = np.zeros(len(present), dtype=bool)
            if cfg.dp.enabled and not self.plan["clip_at_client"]:
                norms = np.linalg.norm(U, axis=1)
                server_clipped = norms > cfg.dp.clip_norm
                U, _, _ = aggregators.norm_bound(U, cfg.dp.clip_norm)
            d = cfg.defense
            agg, info = aggregators.aggregate(
                U, w, rule=cfg.rule, norm_bound_value=cfg.norm_bound, norm_bound_multiplier=d.norm_bound_multiplier,
                trim_fraction=d.trim_fraction, krum_f=d.krum_f, multi_krum_m=d.multi_krum_m,
                gm_max_iter=d.gm_max_iter, gm_tol=d.gm_tol, gm_nu=d.gm_nu)
            for i, s in enumerate(present):
                rows.append({"round": rnd, "site": s, "malicious": int(s in self.malicious), "rule": cfg.rule,
                             "weight": float(w[i]), "norm": float(info["norm"][i]),
                             "clipped": int(info["clipped"][i] or server_clipped[i]),
                             "kept": float(info["kept"][i]), "share": float(info["share"][i]),
                             "flagged": int(info["flagged"][i])})
        if cfg.dp.enabled and self.plan["server_std"] > 0:
            agg = agg + self.rng.normal(0.0, self.plan["server_std"], size=agg.shape)
        return agg, rows


def append_rows(path, rows):
    if not rows:
        return
    import csv
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        if new:
            w.writeheader()
        w.writerows(rows)


def run_local(clients, global_params, rounds, cfg, seed, metrics_dir):
    """The local engine's round loop with the secure pipeline. clients: fedsite.Site objects."""
    sites = [c.name for c in clients]
    sizes = {c.name: c.n_train for c in clients}
    validate(cfg, sites)
    spec = ParamSpec(global_params)
    protos = {c.name: ClientProtocol(cfg, c.name, sites, sizes, seed, spec) for c in clients}
    for c in clients:
        protos[c.name].attach(c)
    server = ServerProtocol(cfg, sites, sizes, seed, spec)
    log_dir = os.path.join(metrics_dir, "fedsec")
    os.makedirs(log_dir, exist_ok=True)
    report = privacy_report(cfg, sites, sizes, rounds, seed)
    write_privacy(os.path.join(log_dir, PRIVACY_FILE), report)
    print(f"[fedsec] {cfg.name}: malicious={report['malicious_sites']} rule={cfg.rule} weights={cfg.weights} "
          f"dp={cfg.dp.mode if cfg.dp.enabled else 'off'} secagg={'on' if cfg.secagg.enabled else 'off'} "
          f"eps_aggregate={report['epsilon_aggregate']:.3g} eps_server={report['epsilon_server']:.3g}", flush=True)
    for rnd in range(rounds):
        msgs, crows = {}, []
        for c in clients:
            p = protos[c.name]
            p.before_round(c, rnd)
            local = copy.deepcopy(c.run_round(rnd, global_params)[0])
            msgs[c.name], row = p.make_message(rnd, global_params, local, c)
            crows.append(row)
        delta, srows = server.aggregate(rnd, msgs)
        global_params = spec.add(global_params, delta)
        append_rows(os.path.join(log_dir, CLIENT_LOG), crows)
        append_rows(os.path.join(log_dir, SERVER_LOG), srows)
    for c in clients:                       # evaluation-only pass over the final aggregate
        c.run_round(rounds, global_params, train=False)
    return global_params
