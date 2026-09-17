# Scenario 2: few events per site (negative result)

Hypothesis: with a survival outcome and only tens of events per site, the
per-SNP Cox regressions that feed the summary-statistic route become unstable
and their standard errors unreliable, so the IVW meta-analysis should degrade
while a pooled Cox model stratified by site, which borrows the baseline hazard
across sites, should hold up.

**The hypothesis is not supported.** At every event count tried, the two
routes give the same bias, spread and coverage.

## Setup

`scripts/rare_events_sweep.py` redraws the ten-site Cox set for each seed with
small populations and short follow-up, then scores both routes against the
true log hazard ratio of 0.3. Run from `data/`:

```bash
uv run python scripts/rare_events_sweep.py --seeds 30                                      # follow-up 1.2
uv run python scripts/rare_events_sweep.py --seeds 30 --followup 0.6 --out results/rare_events_harsh
```

Sites have 300 to 3,000 people and no random censoring; administrative
follow-up sets the event rate. One draw of the milder setting is checked in as
`simulated_data/federated/cox_rare/`, with per-site event counts in its
manifest, and its forest plot is `results/sumstats.cox_rare.png`.

## Results over 30 seeds

| setting | events per site (median, smallest) | route | bias | empirical sd | mean se | RMSE | 95% coverage | mean Q (9 df) |
|---|---|---|---|---|---|---|---|---|
| follow-up 1.2 | 75, 8 | sumstats | +0.025 | 0.123 | 0.110 | 0.124 | 0.87 | 9.1 |
| | | pooled | +0.026 | 0.119 | 0.110 | 0.120 | 0.90 | |
| follow-up 0.6 | 27, 2 | sumstats | -0.006 | 0.177 | 0.183 | 0.174 | 0.97 | 9.8 |
| | | pooled | -0.010 | 0.172 | 0.182 | 0.170 | 0.97 | |

Plots: `results/rare_events.png`, `results/rare_events_harsh.png`. Each
shows the 30 seeds' estimates and intervals for both routes side by side.

## Why it does not separate

- **IVW weights absorb the instability.** A per-SNP Cox fit with almost no
  events at a rare allele returns a wild log hazard ratio with a huge
  standard error. Its weight `1 / se^2` is then close to zero, and the
  site's IVW estimate is driven by the SNPs that were estimable. The same
  information is what the pooled first stage effectively uses.
- **The weak-instrument problem is shared.** With 300 to 1,400 people, mean
  F per site falls to 2 to 7. Single-site IVW estimates swing between -1.2
  and +0.8 on the checked-in draw. But the pooled route fits its first stage
  per site too, so it inherits the same weak instruments. Neither route has
  access to a stronger first stage.
- **Heterogeneity Q stays at its expected value**, so the per-site standard
  errors are not systematically too small at these event counts. The Cox
  Wald standard errors from lifelines held up better than anticipated.

## What would separate them

A version of this scenario where the pooled route genuinely has more
information than the sum of the sites. Two candidates:

1. **Shared SNPs across sites.** Then the pooled first stage has the whole
   sample's strength, mean F scales with total n, and the weak-instrument
   lean disappears from the pooled route only. The sumstats route can match
   this only if sites share per-SNP effects for a meta-GWAS first.
2. **Instrument selection at each site** (scenario 3), where small sites
   select inflated instruments and the pooled data can select on a held-out
   split.
