"""NVFlare client script (Client API). All per-site logic lives in fedsite.py and
is shared with the in-process local engine; this file only speaks NVFlare.
"""

import argparse

import torch

import nvflare.client as flare
from fedsite import Site


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--metrics_dir", required=True)
    p.add_argument("--method", default="2sri", choices=["naive", "2sri", "2sps"])
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--batch_size", type=int, default=0, help="0 = task default")
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    flare.init()
    torch.manual_seed(args.seed)
    site = Site(flare.get_site_name(), args.data_dir, args.metrics_dir, args.method,
                args.epochs, args.lr, args.batch_size, args.test_size, args.seed)
    print(site.describe(), flush=True)

    while flare.is_running():
        input_model = flare.receive()
        params, global_m = site.run_round(input_model.current_round, input_model.params)
        flare.send(flare.FLModel(
            params=params,
            params_type="FULL",
            metrics={k: float(v) for k, v in global_m.items()},
            meta={"NUM_STEPS_CURRENT_ROUND": site.n_train},
        ))


if __name__ == "__main__":
    main()
