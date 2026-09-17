# Data-generating model for MR benchmarks

## Base model

For each individual, draw instruments and two traits:

```
G_j   ~ Binomial(2, p_j)                       j = 1..J SNPs (instruments)
U     ~ N(0, 1)                                unmeasured confounder
X     = sum_j beta_j G_j + gamma_x U + e_x     exposure
Y     = theta X + sum_j alpha_j G_j + gamma_y U + e_y    outcome
```

`theta` is the causal effect the MR method should recover.

## Knobs to vary

- **Pleiotropy** via the `alpha_j`.
  - Zero: valid instruments.
  - Balanced (uncorrelated horizontal pleiotropy, UHP): `alpha_j` with mean zero.
  - Directional: nonzero mean.
  - Correlated (CHP): `alpha_j` correlated with `beta_j`, usually through a
    shared confounder. Violates InSIDE and breaks MR-Egger.
- **Instrument strength** via `beta_j` and `J`. Report the mean F statistic
  and push it into the weak regime (F near 10).
- **Sample design.**
  - One-sample: `beta` and `alpha` estimated in the same people.
  - Two-sample: disjoint exposure and outcome samples.
  - Partial overlap is its own bias.
  - Winner's curse: instruments selected in the same sample used to estimate
    them.
- **Confounding** via `gamma_x`, `gamma_y`. Sets the bias of the naive
  observational regression that MR should remove.
- **Outcome type.** Continuous, or binary through a liability threshold or
  logistic link. Binary outcomes bring non-collapsibility.
- **LD** among instruments, and imprecise LD estimation from a reference panel.

## What to report per method

Bias, coverage and power of `theta_hat` across replicates, for each method
(IVW, MR-Egger, weighted median, MR-PRESSO, and so on).

## Where a pedigree simulator is needed

Population MR is biased by three things that sibling-difference MR removes:
population stratification, assortative mating, and dynastic effects
(parental genotype affecting offspring outcome through the environment).
Simulating those requires real transmission through a pedigree. None of the
surveyed packages do this. See `package-survey.md`.
