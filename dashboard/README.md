# FLAMINGO dashboard

An interactive front end for the simulation and MR scripts in `data/scripts/`:
set the parameters of a synthetic federation, generate it, and run the
Mendelian randomization chain over it.

```bash
cd dashboard
uv sync
uv run streamlit run app.py
```

## What it does

**1 · Data generation** exposes every argument of
`simulate_federated_sites.py` — causal shape and effect sizes, number of sites,
SNPs and population sizes, between-site heterogeneity in instrument strength and
confounding, survival settings, seed — then shows the resulting site manifest,
size distribution and a preview of any site's individual-level data.

**2 · Experiments & results** picks one of those datasets, chooses how to run
the federated workflow over it, runs the chain and compares every estimator:

| step | script | runs for |
|---|---|---|
| Simulate sites | `simulate_federated_sites.py` | all shapes |
| Federated learning | `federated_learning/job.py` | when enabled |
| Conventional MR | `federated_summary_mr.py` | all shapes |
| Non-linear MR | `federated_nonlinear_mr.py` | `quadratic`, `threshold` |

The federated step runs **before** the two MR steps on purpose: both of them
overlay the federated curves they find under the run's `fl/` directory, so this
ordering puts every estimator on the same forest and dose-response plots. The
step runs in `federated_learning/`'s own environment, since it needs torch and
NVFlare; set `FLAMINGO_FL_PYTHON` to point somewhere other than
`federated_learning/.venv/bin/python`. If that interpreter is missing the tab
says so and the MR steps still run.

Its `local` engine reproduces FedAvg's arithmetic in-process in seconds and is
the default; `nvflare` stands up the real simulated federation and costs about
40 s per job.

The results lead with the headline figure — the dose-response curve for the
shapes that have a θ2 to recover, the forest plot otherwise — since that is the
one plot carrying every estimator against the truth. Below it, the **All
estimators** table is the numeric comparison: one row per estimator with what
each one is allowed to see, from per-SNP summary statistics through model
updates to pooled individual rows. The rows needing pooled rows are benchmarks a
real federation could not run. Every figure and table produced by the chain then
follows under **All outputs**, each under its own heading with its download.

## How runs are stored

A run is keyed by a hash of its parameters and written to
`data/results/dashboard_runs/<run_id>/`:

```
params.json             the canonical parameters behind the hash
experiment.json         how the chain was run over that data
sites/<shape>/          site CSVs, truth JSON and the manifest
results/                sumstats.<shape>.{csv,png}, nonlinear.<shape>.png
fl/                     federated results: <method>/<shape>/{curves,metrics}.csv + plots
fl_workspace/           the federated run's scratch space
logs/<step>.log         each step's stdout
```

Selecting a parameter set that has already been run brings its results back
without recomputing; **Force re-run** in the sidebar overrides that.

Reuse is checked two ways, because output paths alone are not enough. A step
records the settings it ran under (`logs/<step>.signature.json`), so changing
the federated rounds or engine — which writes the same file names — re-runs it.
And once any step actually runs, every step after it re-runs too, since they
consume its output.

Two details are load-bearing:

- The sites directory is named after the shape, because
  `federated_summary_mr.py` derives the federated dataset name from that
  directory's name.
- Both MR scripts are pointed at the run's own empty `fl/` directory, so the
  NVFlare curves committed under `federated_learning/results/` cannot be
  overlaid on a run they do not belong to. To compare against an existing
  federated run deliberately, point the step at that root instead.

## Theme

The palette in [`.streamlit/config.toml`](.streamlit/config.toml) is taken from
the pipeline figure in the repository README and from the logo: a pale pink
page, white cards, rose and coral accents. The figure's legend carries meaning,
and the key-results cards reuse it — pink for what travels between sites, a
dashed edge for the benchmark that needs pooled individual rows, coral for the
confounded estimate. Figures produced by `data/scripts/` keep their own colours,
since they are the published results.

## Adding a step

Steps are declared in [`runner.py`](runner.py) and run as subprocesses, so a new
one needs no changes to the app. A `Step` carries its own interpreter and
working directory, a function building its argv, the files it produces, and
optionally a `signature` — the settings a stored result must have been produced
under to count as reusable. The federated step is the worked example:

```python
Step(
    key="federated",
    label="Federated learning (NVFlare FedAvg)",
    script=FL_DIR / "job.py",
    python=FL_PYTHON,
    cwd=FL_DIR,
    argv=_federated_argv,
    outputs=_federated_outputs,
    applies=lambda p: bool(p.get("run_federated")) and bool(p.get("fl_methods")),
    signature=lambda p: {k: p[k] for k in
                         ("fl_methods", "fl_engine", "fl_rounds", "fl_epochs")},
)
```

Parameters that identify the *dataset* belong in `DEFAULTS` and feed the run id;
parameters that only change *how* it is run belong in `EXPERIMENT_DEFAULTS`,
which deliberately stay out of the hash so one dataset can be run several ways.
