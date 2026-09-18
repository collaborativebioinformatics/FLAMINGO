"""Run the FLAMINGO simulation and MR scripts on behalf of the dashboard.

Each step is one of the existing data/scripts/*.py entry points, invoked as a
subprocess. The dashboard never imports the science code, so a step that fails
shows up as a failed step with its traceback rather than a broken app, and the
scripts stay usable on their own from the command line.

A run is keyed by a hash of its parameters and written under
data/results/dashboard_runs/<run_id>/, so repeating a parameter set is free and
two runs can be compared side by side.

Adding a step (for example the NVFlare job in federated_learning/) means
appending a Step to PIPELINE: give it the interpreter and working directory it
needs, a function building its argv, and the files it produces.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "data"
SCRIPTS = DATA_DIR / "scripts"
FL_DIR = REPO / "federated_learning"
RUNS_ROOT = DATA_DIR / "results" / "dashboard_runs"

# The dashboard environment carries the data/ dependencies, so its own
# interpreter runs the scripts. Point this at data/.venv/bin/python to keep the
# two environments separate.
DATA_PYTHON = os.environ.get("FLAMINGO_DATA_PYTHON", sys.executable)

# The repository is one environment, so every step runs in this interpreter.
# Both overrides remain for pointing a step at a separate environment.
FL_PYTHON = os.environ.get("FLAMINGO_FL_PYTHON", sys.executable)
FL_METHODS = ("2sri", "fed2sls")   # the models that run through federated_learning/job.py

# The four model outputs the experiment tab offers, all on by default. The two
# federated ones need the job.py step; the other two come from federated_summary_mr.py.
MODELS = {
    "pooled": "Concatenated 2SLS (pooled individual rows, benchmark)",
    "sumstats": "Summary statistics · per-SNP IVW meta-analysis",
    "fed2sls": "Federated MR · Fed-2SLS (exact 2SLS from summed statistics)",
    "2sri": "Federated MR · Fed-2SRI (control function, FedAvg-trained)",
}
FOREST_ROWS = {"pooled": "pooled", "sumstats": "sumstats", "fed2sls": "fed2sls", "2sri": "federated"}

SHAPES = ("linear", "quadratic", "threshold", "cox")
CURVED_SHAPES = ("quadratic", "threshold")  # the shapes with a theta2 to recover

# Options that change what is run over a dataset, rather than the dataset itself,
# so they deliberately stay out of the run id.
EXPERIMENT_DEFAULTS: dict = {
    "models": list(MODELS),
    "fl_engine": "local",
    "fl_rounds": 5,
    "fl_epochs": 2,
    # bootstrap replicates for the Fed-2SRI confidence band (federated_learning/src/bootstrap.py); 0 = off
    "fl_bootstrap": 200,
}

DEFAULTS: dict = {
    "shape": "quadratic",
    "n_sites": 10,
    "n_snps": 20,
    "pop_min": 1_000,
    "pop_max": 10_000,
    "theta1": 0.3,
    "theta2": 0.15,
    "h2x_mean": 0.10,
    "h2x_kappa": 40.0,
    "gamma_mean": 0.3,
    "gamma_kappa": 20.0,
    "censor_frac": 0.3,
    "followup": 15.0,
    "seed": 1,
}


def canonical(params: dict) -> dict:
    """Drop the parameters the chosen shape ignores, so that changing an
    irrelevant slider does not invalidate a run."""
    p = {k: params.get(k, v) for k, v in DEFAULTS.items()}
    if p["shape"] not in CURVED_SHAPES:
        p.pop("theta2")
    if p["shape"] != "cox":
        p.pop("censor_frac")
        p.pop("followup")
    return p


def full(params: dict) -> dict:
    """Data parameters, which identify the run, plus experiment options, which do not."""
    p = canonical(params)
    for key, default in EXPERIMENT_DEFAULTS.items():
        p[key] = params.get(key, default)
    p["fl_methods"] = [m for m in FL_METHODS if m in p["models"]]   # derived: which models job.py runs
    p["run_federated"] = bool(p["fl_methods"])
    return p


def run_id(params: dict) -> str:
    blob = json.dumps(canonical(params), sort_keys=True).encode()
    return hashlib.sha1(blob).hexdigest()[:12]


@dataclass(frozen=True)
class RunPaths:
    """Layout of one run directory."""

    root: Path

    @property
    def results(self) -> Path:
        return self.root / "results"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def fl(self) -> Path:
        """Federated-learning results root for this run.

        Both MR scripts look up NVFlare curves under <root>/<method>/<dataset>/,
        and pointing them here rather than at federated_learning/results/ keeps a
        run self-contained: an unrelated committed NVFlare run cannot leak into
        this run's plots. It stays empty until an FL step writes into it.
        """
        return self.root / "fl"

    @property
    def sites(self) -> Path:
        """Parent of the per-shape site directories; job.py's --fed_dir."""
        return self.root / "sites"

    def sites_for(self, shape: str) -> Path:
        """Site CSVs for `shape`.

        The directory is named after the shape on purpose: federated_summary_mr.py
        derives the FL dataset name from the sites directory's name.
        """
        return self.sites / shape

    def mkdirs(self) -> None:
        for d in (self.results, self.logs, self.fl):
            d.mkdir(parents=True, exist_ok=True)


def paths_for(params: dict) -> RunPaths:
    return RunPaths(RUNS_ROOT / run_id(params))


@dataclass(frozen=True)
class Step:
    key: str
    label: str
    script: Path
    argv: Callable[[dict, RunPaths], list]
    outputs: Callable[[dict, RunPaths], dict]
    applies: Callable[[dict], bool] = lambda p: True
    # Settings that a stored result must have been produced under to be reusable.
    # Output paths alone are not enough: rerunning with more rounds writes the
    # same file names, so without this a changed setting would look cached.
    signature: Callable[[dict], dict] = lambda p: {}
    # Runs before the cache check and the script (e.g. to clear outputs of a deselected model).
    before: Callable[[dict, RunPaths], None] = lambda p, paths: None
    python: str = field(default=DATA_PYTHON)
    cwd: Path = field(default=DATA_DIR)


def _simulate_argv(p: dict, paths: RunPaths) -> list:
    argv = [
        "--shape", p["shape"],
        "--n-sites", p["n_sites"],
        "--n-snps", p["n_snps"],
        "--pop-min", p["pop_min"],
        "--pop-max", p["pop_max"],
        "--theta1", p["theta1"],
        "--h2x-mean", p["h2x_mean"],
        "--h2x-kappa", p["h2x_kappa"],
        "--gamma-mean", p["gamma_mean"],
        "--gamma-kappa", p["gamma_kappa"],
        "--seed", p["seed"],
        "--out", paths.sites_for(p["shape"]),
    ]
    if p["shape"] in CURVED_SHAPES:
        argv += ["--theta2", p["theta2"]]
    if p["shape"] == "cox":
        argv += ["--censor-frac", p["censor_frac"], "--followup", p["followup"]]
    return argv


def _federated_argv(p: dict, paths: RunPaths) -> list:
    argv = [
        "--dataset", p["shape"],
        # job.py joins fed_dir with the dataset name, which is the shape
        "--fed_dir", paths.sites,
        "--results_root", paths.fl,
        "--workspace", paths.root / "fl_workspace",
        "--engine", p["fl_engine"],
        "--rounds", p["fl_rounds"],
        "--epochs", p["fl_epochs"],
    ]
    for method in p["fl_methods"]:
        argv += ["--method", method]
    if p["shape"] in CURVED_SHAPES:
        # Fed-2SLS on [X, X^2], so it reports theta2 next to the concatenated quadratic fit
        argv += ["--fed2sls_basis", "quadratic"]
    if p.get("fl_bootstrap") and "2sri" in p["fl_methods"]:
        argv += ["--bootstrap", p["fl_bootstrap"]]
    return argv


# One name per federated method, used for the output headings and the estimator table.
# Each says whether it is MR, which MR design, and how it is federated.
METHOD_LABELS = {
    "2sri": "Federated MR · Fed-2SRI (control function, FedAvg-trained)",
    "fed2sls": "Federated MR · Fed-2SLS (exact 2SLS from summed statistics)",
}


def method_label(method: str) -> str:
    return METHOD_LABELS.get(method, f"Federated · {method}")


def _federated_clear_stale(p: dict, paths: RunPaths) -> None:
    """Drop results of federated methods that are no longer selected. job.py only writes the
    methods it runs, and the overview plot and the forest's curve lookup read whatever is on
    disk, so a deselected method would otherwise linger from an earlier run."""
    import shutil
    for method in ("naive", "2sri", "2sps", "fed2sls"):
        if method not in p["fl_methods"]:
            shutil.rmtree(paths.fl / method, ignore_errors=True)


def _federated_outputs(p: dict, paths: RunPaths) -> dict:
    out = {}
    for method in p["fl_methods"]:
        out[method_label(method)] = paths.fl / method / p["shape"] / "fitted_curve.png"
    out["Federated · all methods"] = paths.fl / "fitted_curves_all.png"
    return out


def _sumstats_stem(p: dict, paths: RunPaths) -> Path:
    return paths.results / f"sumstats.{p['shape']}"


def _nonlinear_png(p: dict, paths: RunPaths) -> Path:
    return paths.results / f"nonlinear.{p['shape']}.png"


PIPELINE: tuple[Step, ...] = (
    Step(
        key="simulate",
        label="Simulate sites",
        script=SCRIPTS / "simulate_federated_sites.py",
        argv=_simulate_argv,
        outputs=lambda p, paths: {
            "Site manifest": paths.sites_for(p["shape"]) / "manifest.csv",
            "Site manifest (json)": paths.sites_for(p["shape"]) / "manifest.json",
        },
    ),
    # Before the MR steps on purpose: both of them overlay the federated curves
    # found under the run's fl/ directory, so running this first puts every
    # estimator on the same forest and dose-response plots.
    Step(
        key="federated",
        label="Federated learning (Fed-2SRI via FedAvg; Fed-2SLS exact 2SLS)",
        script=FL_DIR / "job.py",
        python=FL_PYTHON,
        cwd=FL_DIR,
        argv=_federated_argv,
        outputs=_federated_outputs,
        before=_federated_clear_stale,
        signature=lambda p: {**{k: p[k] for k in ("fl_methods", "fl_engine", "fl_rounds", "fl_epochs",
                                                   "fl_bootstrap")},
                             "fed2sls_basis": "quadratic" if p["shape"] in CURVED_SHAPES else "linear"},
        applies=lambda p: bool(p["fl_methods"]),
    ),
    Step(
        key="summary_mr",
        label="Conventional MR (per-site sumstats + IVW meta-analysis)",
        script=SCRIPTS / "federated_summary_mr.py",
        argv=lambda p, paths: [
            "--shape", p["shape"],
            "--sites", paths.sites_for(p["shape"]),
            "--out", _sumstats_stem(p, paths),
            "--fl-results", paths.fl,
            "--rows", *[FOREST_ROWS[m] for m in MODELS if m in p["models"]],
        ],
        signature=lambda p: {"models": [m for m in MODELS if m in p["models"]]},
        outputs=lambda p, paths: {
            "Forest plot": Path(f"{_sumstats_stem(p, paths)}.png"),
            "Per-site estimates": Path(f"{_sumstats_stem(p, paths)}.csv"),
        },
    ),
    Step(
        key="nonlinear_mr",
        label="Non-linear MR (pooled quadratic 2SLS vs the sumstats line)",
        script=SCRIPTS / "federated_nonlinear_mr.py",
        argv=lambda p, paths: [
            "--shape", p["shape"],
            "--sites", paths.sites_for(p["shape"]),
            "--out", _nonlinear_png(p, paths),
            "--federated", paths.fl,
            # the Fed-2SRI curve is the one federated curve on the dose-response plot
            "--federated_methods", *(["2sri"] if "2sri" in p["models"] else []),
        ],
        outputs=lambda p, paths: {"Dose–response curve": _nonlinear_png(p, paths)},
        applies=lambda p: p["shape"] in CURVED_SHAPES,
        signature=lambda p: {"federated_methods": ["2sri"] if "2sri" in p["models"] else []},
    ),
)

STEPS = {s.key: s for s in PIPELINE}

# Bump when the look of a figure changes (colours, labels, layout) so that cached
# runs re-render: every plotting step folds it into its signature.
FIGURE_VERSION = 3
for _step in PIPELINE:
    if _step.key != "simulate":
        _sig = _step.signature
        object.__setattr__(_step, "signature", (lambda p, _s=_sig: {**_s(p), "figure_version": FIGURE_VERSION}))

# What the results page leads with: the dose-response curve where the shape has
# one to recover, otherwise the forest plot.
HERO_OUTPUTS = (("nonlinear_mr", "Dose–response curve"), ("summary_mr", "Forest plot"))


def hero_output(params: dict, paths: RunPaths):
    """(step, label, path) for the headline figure, or None if it is not there yet."""
    p = full(params)
    available = {s.key for s in steps_for(p)}
    for key, label in HERO_OUTPUTS:
        if key not in available:
            continue
        path = STEPS[key].outputs(p, paths).get(label)
        if path and path.exists():
            return STEPS[key], label, path
    return None


def steps_for(params: dict) -> list[Step]:
    p = full(params)
    return [s for s in PIPELINE if s.applies(p)]


@dataclass
class StepResult:
    key: str
    ok: bool
    returncode: int
    command: str
    output: str
    seconds: float
    cached: bool = False


def signature_file(step: Step, paths: RunPaths) -> Path:
    return paths.logs / f"{step.key}.signature.json"


def is_complete(step: Step, params: dict, paths: RunPaths) -> bool:
    p = full(params)
    if not all(path.exists() for path in step.outputs(p, paths).values()):
        return False
    wanted = step.signature(p)
    if not wanted:
        return True
    stored = signature_file(step, paths)
    return stored.exists() and json.loads(stored.read_text()) == wanted


def execute(step: Step, params: dict, paths: RunPaths, force: bool = False) -> StepResult:
    """Run one step, or replay its stored log when its outputs are already there."""
    params = full(params)
    paths.mkdirs()
    log = paths.logs / f"{step.key}.log"
    params = full(params)          # every callback sees the derived keys (fl_methods, run_federated)
    step.before(params, paths)     # also when cached: stale outputs from other selections must go
    if not force and is_complete(step, params, paths) and log.exists():
        return StepResult(step.key, True, 0, "", log.read_text(), 0.0, cached=True)

    argv = [step.python, str(step.script)] + [str(a) for a in step.argv(params, paths)]
    command = " ".join(argv)
    start = time.perf_counter()
    try:
        proc = subprocess.run(argv, cwd=step.cwd, capture_output=True, text=True)
    except OSError as exc:
        # a missing interpreter or script: report it as a failed step rather than
        # letting it take the app down
        message = (f"could not start the step: {exc}\n\n"
                   f"interpreter: {step.python}\nscript: {step.script}\n"
                   f"working directory: {step.cwd}")
        log.write_text(message)
        return StepResult(step.key, False, -1, command, message, time.perf_counter() - start)
    seconds = time.perf_counter() - start
    output = proc.stdout + (f"\n{proc.stderr}" if proc.stderr else "")
    log.write_text(output)
    if proc.returncode == 0:
        signature_file(step, paths).write_text(json.dumps(step.signature(params), indent=1))
    return StepResult(step.key, proc.returncode == 0, proc.returncode, command, output, seconds)


def save_params(params: dict, paths: RunPaths) -> None:
    """params.json identifies the dataset; experiment.json records how it was run."""
    paths.mkdirs()
    (paths.root / "params.json").write_text(json.dumps(canonical(params), indent=1))
    experiment = {k: full(params)[k] for k in EXPERIMENT_DEFAULTS}
    (paths.root / "experiment.json").write_text(json.dumps(experiment, indent=1))


def fl_available() -> bool:
    """Whether the federated step can run: its interpreter exists and has NVFlare."""
    if not Path(FL_PYTHON).exists():
        return False
    probe = subprocess.run([FL_PYTHON, "-c", "import nvflare, torch"],
                           capture_output=True, text=True)
    return probe.returncode == 0


def list_runs() -> list[dict]:
    """Previous runs, newest first, for the run picker."""
    runs = []
    for d in sorted(RUNS_ROOT.glob("*/params.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        runs.append({"run_id": d.parent.name, "params": json.loads(d.read_text()),
                     "modified": d.stat().st_mtime})
    return runs


def data_assets() -> list[dict]:
    """Simulated datasets that experiments can be run against, newest first.

    A run only counts once its sites are on disk, so this is the set of things
    stage 2 can be pointed at.
    """
    assets = []
    for params_file in RUNS_ROOT.glob("*/params.json"):
        params = json.loads(params_file.read_text())
        paths = RunPaths(params_file.parent)
        manifest_file = paths.sites_for(params["shape"]) / "manifest.json"
        if not manifest_file.exists():
            continue
        manifest = json.loads(manifest_file.read_text())
        assets.append({
            "run_id": paths.root.name,
            "params": params,
            "paths": paths,
            "shape": params["shape"],
            "n_sites": len(manifest["sites"]),
            "n": sum(site["n"] for site in manifest["sites"]),
            "modified": manifest_file.stat().st_mtime,
        })
    return sorted(assets, key=lambda a: a["modified"], reverse=True)


def asset_label(asset: dict) -> str:
    p = asset["params"]
    theta = f"θ1={p['theta1']}" + (f", θ2={p['theta2']}" if "theta2" in p else "")
    return (f"{asset['shape']} · {asset['n_sites']} sites · {asset['n']:,} individuals · "
            f"{theta} · seed {p['seed']} · {asset['run_id']}")


_NUM = r"(-?\d+\.?\d*)"
_PATTERNS = {
    "meta_ivw": rf"meta IVW\s+{_NUM}\s+se\s+{_NUM}",
    "pooled": rf"pooled (?:2SLS|2SPS Cox)\s+{_NUM}\s+se\s+{_NUM}",
    # `federated Fed-2SLS: theta1  se1 [ theta2  se2]   (exact, with CI)`; the pair is absent for the linear basis
    "fed2sls": rf"federated Fed-2SLS:\s+{_NUM}\s+{_NUM}(?:\s+{_NUM}\s+{_NUM})?\s+\(exact",
    "heterogeneity_q": rf"heterogeneity Q\s+{_NUM}",
    "pooled_quadratic": rf"concatenated quadratic 2SLS:\s+theta1\s+{_NUM}\s+\({_NUM}\)\s+theta2\s+{_NUM}\s+\({_NUM}\)",
}


def parse_federated(output: str) -> dict:
    """Federated curve parameters that federated_summary_mr.py reports per method.

    The line looks like `federated NVFlare 2sri : 0.295  0.164`; the second
    number is absent for the shapes with no theta2 to recover.
    """
    found = {}
    for method, numbers in re.findall(r"federated NVFlare\s+(\w+)\s*:\s*([-\d. ]+?)\s{2,}\(", output):
        found[method] = tuple(float(v) for v in numbers.split())
    return found


def parse_federated_se(output: str) -> dict:
    """Bootstrap standard errors of those parameters, when the federated step ran --bootstrap.

    The same line then reads `federated NVFlare 2sri : 0.295  0.164   (from curve,
    bootstrap B=200: se 0.012  0.020; 95% CI [...] [...])`.
    """
    found = {}
    for method, numbers in re.findall(r"federated NVFlare\s+(\w+)\s*:.*?bootstrap B=\d+: se ([-\d. ]+?);", output):
        found[method] = tuple(float(v) for v in numbers.split())
    return found


def parse_summary(output: str) -> dict:
    """Pull the headline estimates out of federated_summary_mr.py's printout.

    Best-effort: a line that is missing or reworded is simply absent from the
    result, and the caller falls back to the full log, which is always shown.
    """
    found = {}
    for name, pattern in _PATTERNS.items():
        m = re.search(pattern, output)
        if m:
            found[name] = tuple(float(g) for g in m.groups() if g is not None)
    target = re.search(r"target \((.+?)\)\s+heterogeneity", output)
    if target:
        found["target_label"] = target.group(1)
    return found
