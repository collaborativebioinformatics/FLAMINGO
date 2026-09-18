"""NVFlare client for Fed-2SLS. Builds this site's design once, then answers the
controller's two tasks: its sufficient statistics, and (given theta) its H.

The individual-level data never leave: what is sent is what
flamingo_fedmr.site_stats / site_robust_stats return.
"""

import argparse
import os

import nvflare.client as flare
from nvflare.app_common.abstract.fl_model import FLModel, ParamsType

import flamingo_fedmr as fm
from flamingo_fedmr import protocols as P


def build_design(site: fm.SiteData, spec: dict[str, object], site_index: int) -> fm.Design:
    """The site-local part of the chosen protocol. The shared-instrument quadratic and
    cross-fit variants need a global first stage and are not wired through NVFlare here.
    Cross-fit folds use seed + site_index, exactly as LocalFirstStageFedMR.run does."""
    basis, k = spec["basis"], int(spec.get("crossfit", 0))
    if spec["protocol"] == "shared":
        if basis != "linear" or k:
            raise SystemExit("NVFlare transport implements SharedInstrument with basis=linear, no cross-fit")
        return P.design_shared(site)
    if k:
        xhat = P.crossfit_xhat_local(site, P.fold_ids(site.n, k, int(spec.get("seed", 0)) + site_index), k)
        first_stage = P.local_first_stage_diagnostics(site)
    else:
        xhat = P.local_first_stage(site)
        first_stage = P.local_first_stage_diagnostics(site, xhat)
    return P.design_generated(site, xhat, basis, first_stage)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--sites", required=True, help="comma-separated site names in the order the job uses")
    args = p.parse_args()

    flare.init()
    name = flare.get_site_name()
    site_index = args.sites.split(",").index(name)
    site = fm.load_site_csv(os.path.join(args.data_dir, f"{name}.csv"), name=name)
    design = None
    while flare.is_running():
        message = flare.receive()
        task = message.meta.get("task")
        if design is None:
            design = build_design(site, message.meta, site_index)
            print(f"[{name}] n={site.n} design Z{design.Z.shape} W{design.W.shape}", flush=True)
        if task == "stats":
            st = fm.site_stats(design)
            reply = FLModel(params=st.arrays(), params_type=ParamsType.FULL, meta={"fed2sls": st.meta()})
        elif task == "robust":
            H = fm.site_robust_stats(design, message.meta["theta"])
            reply = FLModel(params={"H": H}, params_type=ParamsType.FULL,
                            meta={"fed2sls": {"site": name, "z_names": list(design.z_names)}})
        else:
            raise SystemExit(f"unknown task {task!r}")
        print(f"[{name}] sent {task}", flush=True)
        flare.send(reply)


if __name__ == "__main__":
    main()
