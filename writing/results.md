# Results

## FedMR equals the pooled analysis

On the ten-site sets (N = 55,182), the exact federated 2SLS reproduces the
concatenated fit in every case:

| set | FedMR θ1 (SE, robust SE) | concatenated θ1 (SE) | max difference |
|---|---|---|---|
| linear | 0.317 (0.0145, 0.0145) | 0.317 (0.0145) | 3e-16 |
| quadratic | 0.318 (0.0148, 0.0149) | 0.318 (0.0148) | 2e-16 |
| threshold | 0.224 (0.0145, 0.0145) | 0.224 (0.0145) | 2e-16 |
| U-shape | 0.018 (0.0148, 0.0149) | 0.018 (0.0148) | 1e-16 |
| linear, shared SNPs | 0.306 (0.0142, 0.0142) | 0.306 (0.0142) | 0 |

The quadratic basis recovers the curve with the pooled covariance: on the
quadratic set θ1 = 0.317 (0.0145), θ2 = 0.149 (0.031) against a truth of
0.30 and 0.15, identical to the concatenated quadratic 2SLS; on the
threshold set θ2 = -0.053 (0.031) records the saturation that the
per-SNP summary statistics cannot see. Run through NVFlare with one client
per site, the federation returns the same numbers to 1e-16 in two rounds
and without training.

The three federated routes now on every forest plot answer different
questions. Per-SNP summary statistics give the average slope with no loss
of precision but no curvature. The FedAvg 2SRI network gives a flexible
curve, including the threshold plateau, but no analytic confidence
interval and a slope that drifts from the truth. FedMR gives the pooled
estimate and interval for a specified basis, in one or two rounds.

## Seed sweep

*Sweep results pending: the 100-seed run of `scripts/fedmr_sweep.py` with the corrected estimand is in progress; this section is filled from `results/fedmr_sweep_summary.csv` when it finishes.*
