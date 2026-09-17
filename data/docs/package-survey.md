# Existing MR simulators

Surveyed 2026-09-17.

## Summary-statistic and individual-level MR

- **simmrd** (Lorincz-Comi, Yang, Zhu; Genetic Epidemiology). Purpose-built
  for evaluating MR methods. Horizontal pleiotropy (UHP and CHP), sample
  overlap, weak instruments, correlated instruments, imprecise LD, winner's
  curse, multivariable MR. Output is summary statistics only.
  https://github.com/noahlorinczcomi/simmrd
  Reviewed in detail in `simmrd-review.md`.
- **simulateGP** (Hemani, MRC IEU). Simulates individual-level genotypes,
  exposures and outcomes with a causal structure, then produces summary
  statistics that feed TwoSampleMR. Continuous or binary outcomes.
  https://explodecomputer.github.io/simulateGP/articles/twosamplemr.html
- **GWASBrewer** (Morrison group). Simulates GWAS summary statistics directly
  for multiple traits connected by a causal graph, no individual-level data.
  Fast; names MR method evaluation as a target use.
  https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11656154/

## Within-family MR

No packaged simulator found. The reference point is Brumpton et al. 2020,
Nature Communications, which used forward-in-time simulations with dynastic
effects, assortative mating and stratification to compare unrelated, sibling
and trio MR designs.
https://www.nature.com/articles/s41467-020-17117-4

### What simACE has

- Real genotypes dropped through a pedigree via tskit (gene drop), so
  instruments obey Mendelian transmission.
- Assortative mating (`assort1`, `assort2`).
- Two traits with shared A, C, E components through rA, rC, rE.
- Sibling and parent-offspring pair extraction via pedigree-graph.
- Liability-threshold and survival outcome models.

### What simACE lacks for MR

- No causal effect between traits (only correlation through rA/rC/rE).
- Gene drop is single-trait; A2 stays parametric.
- No per-SNP dosage export; genetic values are summed per person.
- No GWAS step, no exposure/outcome sample split.
- No dynastic effect term.

Roughly four additions would close the gap: a `theta` term in the two-trait
phenotype model, gene drop on both traits with a shared instrument set, a
dosage export at instrument sites, and a GWAS or sibling-difference
summary-statistic step.

## Recommendation

Use simmrd (or simulateGP for individual-level or binary outcomes) for the
pleiotropy and instrument-strength benchmarks. Reserve a simACE extension for
the within-family and assortative-mating questions.
