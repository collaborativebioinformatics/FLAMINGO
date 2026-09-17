# Federated learning with NVFlare

A minimal NVFlare (2.9) FedAvg setup that trains a small non-linear
classifier on the simulated multi-site data in
`../data/simulated_data/federated/<dataset>/`. One simulated client per
`siteNN.csv`.

| File | What |
|---|---|
| `job.py` | Builds the `FedAvgJob`, attaches `src/client.py` to every site, runs the NVFlare simulator, prints per-round tables |
| `src/model.py` | `MLP`: 1 -> 32 -> 32 -> 1 with ReLU (feature `X`, logit of `Y = 1`) |
| `src/client.py` | Client API script: 80/20 stratified split, local training, evaluation, metrics CSV |
| `workspace/` | Simulator output (git-ignored). Per-client metrics in `workspace/metrics/<site>.csv` |

## Run

```bash
uv sync                     # installs nvflare, torch (CPU), pandas, scikit-learn
uv run python job.py        # binary_quadratic_logistic, 10 sites, 5 rounds, 2 local epochs
uv run python job.py --rounds 10 --epochs 3 --lr 0.005
uv run python job.py --dataset quadratic   # continuous Y: classifies Y > 0
```

## What happens each round

1. The server sends the global model to every client.
2. Each client evaluates it on its own 20% test split and prints
   loss, accuracy, precision, recall, F1 and AUC (`[siteNN] round r global-model test: ...`).
3. Each client trains for `--epochs` epochs on its 80% train split (Adam, BCE loss).
4. Each client evaluates the updated model on the same test split and prints it
   (`[siteNN] round r local-model test: ...`).
5. Weights go back to the server, which averages them weighted by train size.

After the simulator finishes, `job.py` prints one table per round and stage
with every client's metrics plus a test-size-weighted mean.

## Notes

- Default dataset is `binary_quadratic_logistic` because its `Y` is already 0/1.
  For the continuous datasets (`linear`, `quadratic`, `threshold`) the client
  turns `Y` into `Y > 0`. The `cox` set has no `Y` column and is not supported.
- Case prevalence is 0.3 and `X` alone carries little signal in these
  simulations, so accuracy sits near the majority-class rate (~0.70) and AUC
  near 0.58. That is a property of the data, not the federation.
