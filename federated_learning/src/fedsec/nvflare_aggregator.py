"""The server half of protocol.py as an NVFlare ModelAggregator, plugged into NVFlare's own FedAvg
controller (nvflare.app_common.workflows.fedavg.FedAvg(aggregator=...)). The controller's round
loop, client sampling and persistence are unchanged; only the combination step is replaced.

Each client sends its fedsec Message in params["fedsec_vec"] plus meta; the aggregator collects
them, runs ServerProtocol.aggregate, and returns the aggregate update as params_type DIFF, which
the controller adds to the global model.

NVFlare rebuilds components from attributes named like the __init__ arguments, so every argument
is a plain builtin (dict / list / str / int) and is stored under its own name.
"""

import os

from nvflare.app_common.abstract.fl_model import FLModel, ParamsType
from nvflare.app_common.aggregators.model_aggregator import ModelAggregator

from .config import from_dict
from .protocol import SERVER_LOG, Message, ParamSpec, ServerProtocol, append_rows


class FedSecAggregator(ModelAggregator):
    def __init__(self, config: dict, sites: list, sizes: dict, param_spec: list, seed: int = 0, log_dir: str = ""):
        super().__init__()
        self.config, self.sites, self.sizes, self.param_spec = config, sites, sizes, param_spec
        self.seed, self.log_dir = seed, log_dir
        self._spec = ParamSpec(spec=param_spec)
        self._server = ServerProtocol(from_dict(config), sites, sizes, seed, self._spec)
        self._msgs = {}

    def accept_model(self, model: FLModel):
        site = (model.meta or {}).get("client_name")
        if site not in self.sizes:
            raise RuntimeError(f"fedsec: result from unknown client {site!r}")
        self._msgs[site] = Message.from_flare(model.params, model.meta)

    def aggregate_model(self) -> FLModel:
        rnd = next(iter(self._msgs.values())).round if self._msgs else -1
        delta, rows = self._server.aggregate(rnd, self._msgs)
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
            append_rows(os.path.join(self.log_dir, SERVER_LOG), rows)
        self._msgs = {}
        return FLModel(params_type=ParamsType.DIFF, params=self._spec.unflatten_numpy(delta))

    def reset_stats(self):
        self._msgs = {}
