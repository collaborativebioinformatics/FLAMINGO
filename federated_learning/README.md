# Federated learning with NVFlare

A minimal NVFlare (2.9) FedAvg setup that trains a small non-linear model of
`X -> outcome` on the simulated multi-site data in
`../data/simulated_data/federated/<dataset>/`, one simulated client per
`siteNN.csv`. The same network and federation run under every outcome model
the simulator produces; only the loss and the metrics change.

| Dataset | Outcome | Task | Loss | Metrics |
|---|---|---|---|---|
| `linear`, `quadratic`, `threshold` | `Y` real-valued | continuous | MSE | mse, mae, r2 |
| `binary_quadratic_logistic` | `Y` in {0, 1} | binary | BCE | accuracy, precision, recall, f1, auc |
| `cox`, `cox_rare` | `time`, `event` | survival | Cox partial likelihood (Breslow) | c_index |

The task is detected from the columns and `manifest.json`, see `src/tasks.py`.

| File | What |
|---|---|
| `job.py` | Builds the `FedAvgJob`, attaches `src/client.py` to every site, runs the simulator, prints per-round tables, writes results and plots |
| `src/model.py` | `MLP`: 1 -> 32 -> 32 -> 1 with ReLU. Its output is predicted Y, a logit, or a log relative hazard depending on the task |
| `src/tasks.py` | The three outcome families: targets, loss, metrics, and the true causal curve from the manifest |
| `src/client.py` | Client API script: 80/20 split, local training, evaluation, fitted-curve recording |
| `src/plots.py` | Per-dataset metric and fitted-curve plots plus the all-datasets overview |
| `results/<dataset>/` | `metrics.csv`, `curves.csv`, `metrics_by_round.{global,local}.png`, `fitted_curve.png` |
| `results/fitted_curves_all.png`, `results/summary.csv` | Overview across datasets |
| `workspace/` | Simulator output (git-ignored) |

## Run

```bash
uv sync                                      # nvflare, torch (CPU), pandas, scikit-learn, lifelines, matplotlib
uv run python job.py --all                   # every dataset: 10 sites, 5 rounds, 2 local epochs
uv run python job.py --dataset cox           # one dataset (repeatable flag)
uv run python job.py --all --rounds 10 --epochs 3 --lr 0.005
uv run python src/plots.py                   # re-render plots from results/ without training
```

## What happens each round

1. The server sends the global model to every client.
2. Each client evaluates it on its own 20% test split and prints the task's
   metrics (`[siteNN] round r global-model test: ...`), and records the
   model's output on a fixed grid of X values.
3. Each client trains for `--epochs` epochs on its 80% train split (Adam).
4. Each client evaluates the updated model on the same test split and prints it.
5. Weights go back to the server, which averages them weighted by train size.

After the simulator finishes, `job.py` prints one table per round and stage
with every client's metrics plus a test-size-weighted mean.

## Reading the fitted curves

`fitted_curve.png` overlays the last-round global model's `f(X)` on the true
causal curve from the manifest and on the pooled binned mean of the outcome.
The network fits `E[outcome | X]`, which includes the confounder `U`'s path
from X to the outcome, so it tracks the binned means and not the causal
curve. That gap is the confounding that the MR analyses in `../data/` are
there to remove; this federation shows what a plain predictive model learns
across sites, not the causal effect. For survival the curve is the learned
log hazard ratio anchored at X = 0; the binary set has no comparable
liability-scale curve, so only the empirical prevalence is shown.

## Notes

- Last-round weighted test metrics for the default run are in `results/summary.csv`.
- `X` alone explains little of the outcome by design (R² near 0.1, AUC near
  0.58, c-index near 0.6), so the binary model rarely predicts a case and its
  accuracy sits at the majority-class rate. That is a property of the data.
- `cox_rare` has 5 to 26 events per test split, so its per-site c-index is noisy.
