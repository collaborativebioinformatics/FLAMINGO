"""Server-side FedMR workflow for NVFlare: sum the sites' statistics, solve, optionally
collect the robust-covariance round. No training, no averaging.

Round 0 ("stats")   broadcast the protocol spec; each site returns SiteStats
                    (arrays in params, column names and roles in meta)
Round 1 ("robust")  broadcast theta by column name; each site returns H = Z' diag(u^2) Z

The arithmetic is flamingo_fedmr's: the same aggregate() and fit() the data
scripts and tests use, so the NVFlare result equals the in-process result.
"""

import json
import os

import numpy as np
from nvflare.app_common.abstract.fl_model import FLModel, ParamsType
from nvflare.app_common.workflows.model_controller import ModelController

import flamingo_fedmr as fm


class FedMRController(ModelController):
    def __init__(self, protocol: str = "local", basis: str = "linear", crossfit: int = 0,
                 robust: bool = True, out_path: str = "", seed: int = 0, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.protocol, self.basis, self.crossfit, self.robust, self.out_path = protocol, basis, crossfit, robust, out_path
        self.seed = seed

    def run(self) -> None:
        spec = {"protocol": self.protocol, "basis": self.basis, "crossfit": self.crossfit, "seed": self.seed}
        self.info(f"FedMR round 0: collecting sufficient statistics ({spec})")
        replies = self.send_model_and_wait(
            task_name="train", data=FLModel(params={}, params_type=ParamsType.FULL, current_round=0, total_rounds=2,
                                             meta={"task": "stats", **spec}))
        parts = [fm.SiteStats.from_transport(r.params, r.meta["fedmr"]) for r in replies]
        stats = fm.aggregate(parts)
        res = fm.fit(stats)
        rounds = 1
        estimates = ", ".join(f"{name} = {res[name]:.6f} ({res.se(name):.6f})" for name in res.w_names)
        self.info(f"FedMR fit over {len(parts)} sites, N = {stats.N}: {estimates}")

        if self.robust:
            self.info("FedMR round 1: robust covariance")
            replies = self.send_model_and_wait(
                task_name="train", data=FLModel(params={}, params_type=ParamsType.FULL, current_round=1,
                                                 total_rounds=2, meta={"task": "robust", "theta": res.theta_by_name(),
                                                                       **spec}))
            H = fm.aggregate_robust([(r.meta["fedmr"]["z_names"], np.asarray(r.params["H"], float)) for r in replies],
                                    stats.layout)
            res.robust_cov = fm.robust_cov(stats, H)
            rounds += 1

        out = {"sites": stats.layout.sites, "rounds": rounds, "spec": spec, **res.to_dict()}
        if self.out_path:
            os.makedirs(os.path.dirname(self.out_path), exist_ok=True)
            with open(self.out_path, "w") as output_file:
                json.dump(out, output_file, indent=1)
            self.info(f"wrote {self.out_path}")
