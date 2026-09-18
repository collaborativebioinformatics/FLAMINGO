# Results

## Fed-2SLS equals the pooled analysis

On the ten-site sets (N = 55,182), the exact federated 2SLS reproduces the
concatenated fit in every case:

| set | Fed-2SLS θ1 (SE, robust SE) | concatenated θ1 (SE) | max difference |
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
interval and a slope that drifts from the truth. Fed-2SLS gives the pooled
estimate and interval for a specified basis, in one or two rounds.

## Seed sweep

Over 100 seeds per setting (`data/scripts/fed2sls_sweep.py`, ten sites of 500
to 5,000 people), Fed-2SLS equalled the pooled fit to 1.7e-15 in every one of
2,200 replicates. Against the pooled-2SLS estimand, the number of sites
(2 to 20), sample-size imbalance (up to 90/10) and allele-frequency shifts
between sites did not separate the federated from the pooled or the
summary-statistics routes. Weak instruments did: at a first-stage F near 4
every one-sample route leaned 0.025 towards the confounded association with
coverage 0.72 to 0.87, while cross-fitted Fed-2SLS, which uses the out-of-fold
genetic prediction as the instrument, had bias within 0.001 and coverage
0.93 to 0.95. With shared SNPs the pooled first stage (F about 150) removed
a 0.010 lean that site-level first stages (F about 15) kept. Under
site-specific causal effects the pooled, federated and site-meta-analysis
routes agreed on the first-stage-weighted mean while per-SNP IVW drifted by
0.014. Pleiotropy of sd 0.02 per allele reduced coverage to 0.78 to 0.87
for every route alike; exact federation does not protect against invalid
instruments.
