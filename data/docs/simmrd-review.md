# simmrd: source-level review

Reviewed 2026-09-17 against the GitHub main branch (last commit 2026-05-17).

## What it is

- R package by Lorincz-Comi, Yang and Zhu. MIT licence.
- About 1000 lines of R in two files (`R/simmrd.R`, `R/set_params.R`).
- Depends on mvnfast, ggplot2, ggpubr.
- Install: `remotes::install_github('noahlorinczcomi/simmrd')`.
- Ships a pixi-managed CLI (`cli/`) that runs Monte Carlo replicates from a
  YAML parameter file and writes an `.rds`.

## API

```r
params <- set_params(...)        # validated defaults
data   <- generate_summary(params)
data   <- generate_individual(params)
```

`load_preset()` reproduces the paper's named scenarios
(bias in {none, UHP, CHP, UHP_CHP, UHP_CHP_WEAK, WEAK}; n in {3e4, 1e5};
snps in {100, 500}; exposures in {1, 3}; overlap in {full, none}).

Key parameters: exposure and outcome GWAS sizes and their overlap, number of
causal SNPs, UHP and CHP counts and variance shares, true causal effects,
confounder U variance shares, LD structure, instrument selection by P-value
threshold (`simtype = "winners"`) or fixed mean F (`simtype = "weak"`),
number of exposures (MVMR).

## Output

Both generators return the same summary-statistic list: `bx`, `bxse`, `by`,
`byse`, `RhoME` (measurement-error correlation), `LDMatrix`, `LDhatMatrix`,
`theta`, `IVtype` (valid / UHP / CHP), plus unstandardised versions.

**`generate_individual()` does not return genotypes or phenotypes.** It
simulates them internally, runs the GWAS, selects instruments, and returns
the summary-stat list. Verified at the `return(out)` in `R/simmrd.R`.

## Limits

- Genotypes are independent binomial draws at a fixed MAF of 0.3, with LD
  imposed by a correlation matrix. No population structure, relatedness or
  families.
- No MR estimators are wrapped. Feed the output to MendelianRandomization,
  TwoSampleMR, or your own code.
- Continuous exposures and outcome only.

## Verdict

Good fit for evaluating summary-statistic MR methods under pleiotropy, weak
instruments, overlap and winner's curse. Wrong tool for individual-level
data, binary outcomes, or family structure.
