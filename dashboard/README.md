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

**2 · Experiments & results** runs the chain over that data and shows the
headline estimates, the forest plot and (for the curved shapes) the
dose-response plot:

| step | script | runs for |
|---|---|---|
| Simulate sites | `simulate_federated_sites.py` | all shapes |
| Conventional MR | `federated_summary_mr.py` | all shapes |
| Non-linear MR | `federated_nonlinear_mr.py` | `quadratic`, `threshold` |

## How runs are stored

A run is keyed by a hash of its parameters and written to
`data/results/dashboard_runs/<run_id>/`:

```
params.json             the canonical parameters behind the hash
sites/<shape>/          site CSVs, truth JSON and the manifest
results/                sumstats.<shape>.{csv,png}, nonlinear.<shape>.png
fl/                     federated-learning results root for this run
logs/<step>.log         each step's stdout
```

Selecting a parameter set that has already been run brings its results back
without recomputing; **Force re-run** in the sidebar overrides that.

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
one needs no changes to the app. Adding the NVFlare job, for example, means
appending a `Step` that carries its own interpreter and working directory:

```python
Step(
    key="federated",
    label="Federated learning (NVFlare FedAvg)",
    script=REPO / "federated_learning" / "job.py",
    python=os.environ.get(
        "FLAMINGO_FL_PYTHON",
        str(REPO / "federated_learning" / ".venv" / "bin" / "python"),
    ),
    cwd=REPO / "federated_learning",
    argv=lambda p, paths: [
        "--dataset", p["shape"], "--method", "2sri",
        "--engine", "local",   # the NVFlare simulator costs ~40 s per job
    ],
    outputs=lambda p, paths: {
        "fitted curve": paths.fl / "2sri" / p["shape"] / "fitted_curve.png",
    },
)
```

That step is not wired up yet: `job.py` resolves its data and results
directories relative to the repository, so it needs flags for both before it can
write into a run directory.
