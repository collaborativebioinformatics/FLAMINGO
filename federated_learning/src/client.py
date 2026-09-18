"""NVFlare client script (Client API). All per-site logic lives in fedsite.py and
is shared with the in-process local engine; this file only speaks NVFlare.

With --fedsec (a JSON file job.py writes when a secure config is active) the
site sends the fedsec protocol message (fedsec/protocol.py) instead of its
full weights; without it the script behaves exactly as before.
"""

import argparse
import json
import os

import torch

import nvflare.client as flare
from fedsite import Site


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--metrics_dir", required=True)
    p.add_argument("--method", default="2sri", choices=["naive", "2sri", "2sps"])
    p.add_argument("--rounds", type=int, required=True,
                   help="training rounds; round == rounds is the evaluation-only pass over the final aggregate")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--batch_size", type=int, default=0, help="0 = task default")
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fedsec", default="", help="secure-federation settings written by job.py (JSON)")
    args = p.parse_args()

    flare.init()
    torch.manual_seed(args.seed)
    site = Site(flare.get_site_name(), args.data_dir, args.metrics_dir, args.method,
                args.epochs, args.lr, args.batch_size, args.test_size, args.seed)
    print(site.describe(), flush=True)

    proto = None
    if args.fedsec:
        from fedsec.config import from_dict
        from fedsec.protocol import ClientProtocol, ParamSpec, append_rows
        with open(args.fedsec) as f:
            fs = json.load(f)
        cfg = from_dict(fs["config"])
        proto = ClientProtocol(cfg, site.name, fs["sites"], fs["sizes"], args.seed, ParamSpec(spec=fs["param_spec"]))
        proto.attach(site)
        # per-site test metrics are a side channel the dp/secagg guarantees do not cover: keep them local
        share_metrics = not (cfg.dp.enabled or cfg.secagg.enabled)
        log_path = os.path.join(args.metrics_dir, "fedsec", f"clients.{site.name}.csv")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    while flare.is_running():
        input_model = flare.receive()
        rnd = input_model.current_round
        train = rnd < args.rounds
        if proto is None:
            params, global_m = site.run_round(rnd, input_model.params, train=train)
            flare.send(flare.FLModel(
                params=params,
                params_type="FULL",
                metrics={k: float(v) for k, v in global_m.items()},
                meta={"NUM_STEPS_CURRENT_ROUND": site.n_train},
            ))
            continue

        proto.before_round(site, rnd)
        global_state = {k: torch.as_tensor(v).detach().clone() for k, v in input_model.params.items()}
        local_state, global_m = site.run_round(rnd, input_model.params, train=train)
        if train:
            msg, row = proto.make_message(rnd, global_state, local_state, site)
            append_rows(log_path, [row])
        else:
            msg = proto.eval_message(rnd)
        params, meta = msg.to_flare()
        meta["NUM_STEPS_CURRENT_ROUND"] = site.n_train
        flare.send(flare.FLModel(
            params=params,
            params_type="FULL",
            metrics={k: float(v) for k, v in global_m.items()} if share_metrics else None,
            meta=meta,
        ))


if __name__ == "__main__":
    main()
