# FLAMINGO dashboard

An interactive front end for the simulation and MR scripts in `data/scripts/`:
set the parameters of a synthetic federation, generate it, and run the
Mendelian randomization chain over it.

```bash
uv sync            # once, at the repository root
cd dashboard
uv run streamlit run app.py
```

## What it does

**1 · Data generation** exposes every argument of
`simulate_federated_sites.py` — causal shape and effect sizes, number of sites,
SNPs and population sizes, between-site heterogeneity in instrument strength and
confounding, survival settings, seed — then shows the resulting site manifest,
size distribution and a preview of any site's individual-level data.

**2 · Experiments & results** picks one of those datasets, chooses how to run
the federated workflow over it, runs the chain and compares the selected
model outputs (concatenated 2SLS, per-SNP summary statistics, Fed-2SLS and
Fed-2SRI; all four on by default):

| step | script | runs for |
|---|---|---|
| Simulate sites | `simulate_federated_sites.py` | all shapes |
| Federated learning (Fed-2SRI via FedAvg; Fed-2SLS exact 2SLS) | `federated_learning/job.py` | when either federated model is selected; Fed-2SLS on continuous shapes |
| Conventional MR | `federated_summary_mr.py` | all shapes |
| Non-linear MR | `federated_nonlinear_mr.py` | `quadratic`, `threshold` |

The federated step runs **before** the two MR steps on purpose: both of them
overlay the federated curves they find under the run's `fl/` directory, so this
ordering puts every estimator on the same forest and dose-response plots. The
repository is one environment, so every step runs in the interpreter the
dashboard itself is running under; `FLAMINGO_DATA_PYTHON` and
`FLAMINGO_FL_PYTHON` still point individual steps elsewhere. If that interpreter
cannot import torch and NVFlare the tab says so and the MR steps still run.

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

**3 · Sensitivity & invariance** asks how much the answer depends on which
sites, which instruments and which fit. It needs no run: it works from the
generated sites (or, before any run, from the committed federation in
`data/simulated_data/federated/<shape>`) and computes everything live from one
small bundle per site — `n`, column means and the site-centred cross products of
`(Y, X, X², x̂, x̂²)`, plus `G′G` and `G′(Y, X)` — so every estimate below is a
sum of per-site matrices, which is what a federation would exchange. Shapes with
a continuous outcome only (linear, quadratic, threshold).

| sub-tab | what it shows |
|---|---|
| Overview | every estimator on one forest (pooled 2SLS, IVW / random-effects / equal-weight meta-analysis, minimax, anchor γ→∞, PULSE, LIML, naive); naive vs 2SLS slope per site |
| Leave sites out | drop-1 forest, influence bubbles (IV vs naive), leave-k-out distributions, cumulative meta-analysis in a chosen order, per-site forest with FE/RE and prediction interval, funnel, pairwise disagreement, weak-instrument filter |
| Regularisation | K-class path OLS → PULSE → 2SLS → LIML with the instrument test along it, ridge on the second stage, site re-weighting by precision^a |
| Robust across sites | anchor regression with the site as anchor (γ path, IV and naive), the minimax estimate no site objects to (Wald fan or contour), V-REx path |
| Invariance & ICP | GMM validity certificate split into within-site (instrument validity) and between-site (same θ) parts; per-SNP cross-site invariance heatmap and greedy invariant-instrument search; classic ICP with sites as environments (with the oracle confounder) next to IV-ICP |

A **what-if** panel perturbs the loaded rows in memory — pleiotropic SNPs, a
site with a deviant effect, an outcome level shift, extra confounding at one
site — so each analysis can be seen catching (or missing) each violation.
Nothing on disk changes. Science lives in `sensitivity.py`, figures in
`charts.py`, the Streamlit layout in `sensitivity_tab.py`.

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
the pipeline figure in the repository README and from the logo. The figure's
legend carries meaning, and the key-results cards reuse it — pink for what
travels between sites, a dashed edge for the benchmark that needs pooled
individual rows, coral for the confounded estimate. Figures produced by
`data/scripts/` keep their own colours, since they are the published results.

Light and dark both ship: `[theme.light]` is the figure's pale pink page and
white cards, `[theme.dark]` rebuilds the same relationships on the logo's
near-black plum, and the viewer's browser or OS setting picks between them.

**`.streamlit/config.toml` has to be committed.** Without it Streamlit falls
back to its own default theme, which follows the browser into dark mode while
the app's CSS keeps painting light-mode cards — the symptom is unreadable
headings on a dark page.

Two things follow from Streamlit exposing no theme CSS variables:

- The custom palette is injected from Python in [`app.py`](app.py) (`PALETTES`,
  `css_for`), keyed on the theme Streamlit reports through `st.context.theme`.
  A `prefers-color-scheme` media query would be wrong whenever the app's own
  theme setting disagrees with the OS.
- Every custom colour must come from a `--fl-*` variable defined in both
  palettes. A hardcoded colour is exactly what breaks in the other mode.

## Adding a step

Steps are declared in [`runner.py`](runner.py) and run as subprocesses, so a new
one needs no changes to the app. A `Step` carries its own interpreter and
working directory, a function building its argv, the files it produces, and
optionally a `signature` — the settings a stored result must have been produced
under to count as reusable. The federated step is the worked example:

```python
Step(
    key="federated",
    label="Federated learning (FedAvg: naive, MR 2SRI / 2SPS; Fed-2SLS exact 2SLS)",
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
