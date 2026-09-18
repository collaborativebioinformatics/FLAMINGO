"""Fed-2SLS on the federated sets: exact federated 2SLS versus the other three families.

For each continuous set this runs, without ever concatenating rows across sites,

    Fed-2SLS             sufficient statistics summed over sites (the flamingo_fedmr package),
                      classical and robust standard errors, first-stage F and partial R^2.
                      The protocol follows the manifest: LocalFirstStage for site-specific
                      SNPs (this repo's default sets), SharedInstrument when shared_snps.
    Fed-2SLS-CF          the same with k-fold cross-fitted X_hat as the generated instrument

and compares them with

    concatenated      one pooled 2SLS on the stacked rows (federated_summary_mr.py)
    site meta         each site's own 2SLS, combined by inverse-variance meta-analysis
    sumstats IVW      per-SNP GWAS effects, per-site IVW, meta-analysis

The headline number is |Fed-2SLS - concatenated|, which must be below 1e-10: the
federated estimator is the pooled estimator, not an approximation of it.

Writes results/fed2sls.<shape>.csv (one row per estimator) that the forest and
dose-response plots pick up, and prints the table.

    uv run python scripts/federated_exact_mr.py --shape quadratic
    uv run python scripts/federated_exact_mr.py --all
"""

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import polars as pl

import flamingo_fedmr as fm

sys.path.insert(0, str(Path(__file__).parent))
from federated_nonlinear_mr import sumstats_slope  # noqa: E402
from federated_summary_mr import multivariate_meta, pooled_2sls, quadratic_2sls  # noqa: E402

FED = Path("simulated_data/federated")
SHAPES = ("linear", "quadratic", "ushape", "threshold")   # prefixes: linear_shared etc. count too


def raw(sites: list[fm.SiteData]) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    return [(s.G, s.X, s.Y) for s in sites]


def pooled_2sls_shared(sites: list[fm.SiteData]) -> tuple[float, float, float]:
    """Reference for shared SNPs: one stacked 2SLS with the common G plus site dummies in Z and W
    (one global first stage with site intercepts). Returns (theta, se, naive)."""
    K = len(sites)
    Z, W, Y = [], [], []
    for k, s in enumerate(sites):
        S = np.zeros((s.n, K))
        S[:, k] = 1
        Z.append(np.column_stack([s.G, S]))
        W.append(np.column_stack([s.X, S]))
        Y.append(s.Y)
    Z, W, Y = np.vstack(Z), np.vstack(W), np.concatenate(Y)
    P = Z @ np.linalg.lstsq(Z, W, rcond=None)[0]
    theta = np.linalg.lstsq(P, Y, rcond=None)[0]
    u = Y - W @ theta
    sigma2 = u @ u / (len(Y) - W.shape[1])
    se = np.sqrt(sigma2 * np.linalg.inv(P.T @ P)[0, 0])
    naive = np.linalg.lstsq(W, Y, rcond=None)[0][0]
    return float(theta[0]), float(se), float(naive)


def site_meta_linear(sites: list[fm.SiteData]) -> tuple[float, float]:
    est, se = zip(*[pooled_2sls([r])[:2] for r in raw(sites)])
    w = 1 / np.array(se) ** 2
    return float(np.sum(w * est) / np.sum(w)), float(np.sqrt(1 / np.sum(w)))


@dataclass(kw_only=True)
class EstimatorRow:
    """One line of the comparison table (results/fed2sls.<shape>.csv)."""
    estimator: str
    theta1: float
    se1: float | None = None
    robust_se1: float | None = None
    theta2: float | None = None
    se2: float | None = None
    robust_se2: float | None = None
    first_stage_F: float | None = None
    partial_r2: float | None = None
    generated_instrument_F: float | None = None
    abs_diff_from_pooled: float | None = None
    rounds: int | None = None
    what_leaves_site: str = ""


def fed2sls_row(
    name: str,
    run: fm.Run,
    diff: float | None = None,
    leaves: str = "",
    theta2: bool = False,
) -> EstimatorRow:
    """Row for a Fed-2SLS run: estimates, both SEs and first-stage diagnostics from the result."""
    r = run.result
    fs = r.diagnostics.first_stage["X"]
    return EstimatorRow(estimator=name, theta1=r["X"], se1=r.se("X"), robust_se1=r.se("X", True),
                        theta2=r["X2"] if theta2 else None, se2=r.se("X2") if theta2 else None,
                        robust_se2=r.se("X2", True) if theta2 else None,
                        first_stage_F=fs["F"], partial_r2=fs["partial_r2"],
                        generated_instrument_F=fs.get("generated_instrument_F"),
                        abs_diff_from_pooled=diff, rounds=run.rounds, what_leaves_site=leaves)


def analyse(shape: str, crossfit: int, out_dir: Path) -> pl.DataFrame:
    folder = FED / shape
    manifest = json.loads((folder / "manifest.json").read_text())
    shared = bool(manifest.get("shared_snps"))
    protocol_class = fm.SharedInstrumentFedMR if shared else fm.LocalFirstStageFedMR
    protocol_name = "SharedInstrument" if shared else "LocalFirstStage"
    sites = fm.load_sites(folder)
    curved = not shape.startswith("linear")
    rows = []

    if shared:
        pooled, pooled_se, naive = pooled_2sls_shared(sites)
        rows.append(EstimatorRow(estimator="concatenated 2SLS (one shared first stage)", theta1=pooled, se1=pooled_se,
                                 what_leaves_site="individual rows"))
        p_site, p_site_se, _ = pooled_2sls(raw(sites))
        rows.append(EstimatorRow(estimator="concatenated 2SLS (per-site first stages)", theta1=p_site, se1=p_site_se,
                                 abs_diff_from_pooled=abs(p_site - pooled),
                                 what_leaves_site="individual rows; a different estimator, for contrast"))
    else:
        pooled, pooled_se, naive = pooled_2sls(raw(sites))
        rows.append(EstimatorRow(estimator="concatenated 2SLS", theta1=pooled, se1=pooled_se,
                                 what_leaves_site="individual rows"))

    run = protocol_class(basis="linear", robust=True).run(sites)
    res = run.result
    rows.append(fed2sls_row(f"Fed-2SLS ({protocol_name})", run, diff=abs(res["X"] - pooled),
                          leaves="centred Z'Z, Z'W, Z'Y, W'W, W'Y, Y'Y, n (+ first-stage RSS); then H for the robust SE"))
    rows.append(fed2sls_row(f"Fed-2SLS-CF ({crossfit} folds)",
                          protocol_class(basis="linear", robust=True, crossfit=crossfit).run(sites),
                          leaves="per-fold first-stage moments, then the same statistics"))

    meta, meta_se = site_meta_linear(sites)
    rows.append(EstimatorRow(estimator="site 2SLS meta-analysis", theta1=meta, se1=meta_se,
                             abs_diff_from_pooled=abs(meta - pooled), what_leaves_site="one estimate and SE per site"))
    slope, slope_se = sumstats_slope(raw(sites))
    rows.append(EstimatorRow(estimator="per-SNP sumstats IVW", theta1=slope, se1=slope_se,
                             abs_diff_from_pooled=abs(slope - pooled), what_leaves_site="per-SNP beta_x, beta_y, SEs"))
    rows.append(EstimatorRow(estimator="naive OLS (no instruments)", theta1=naive))

    if curved and shared:
        print("note: the quadratic comparison uses per-site first stages (quadratic_2sls); skipped for shared sets")
    if curved and not shared:
        theta, cov = quadratic_2sls(raw(sites))
        rows.append(EstimatorRow(estimator="concatenated quadratic 2SLS", theta1=theta[0], se1=np.sqrt(cov[0, 0]),
                                 theta2=theta[1], se2=np.sqrt(cov[1, 1]), what_leaves_site="individual rows"))
        qrun = protocol_class(basis="quadratic", robust=True).run(sites)
        q = qrun.result
        d = max(abs(q["X"] - theta[0]), abs(q["X2"] - theta[1]))
        rows.append(fed2sls_row("Fed-2SLS quadratic", qrun, diff=d, theta2=True,
                              leaves="the same statistics with W = [X, X^2], Z = [xhat, xhat^2], centred within site"))
        rows.append(fed2sls_row(f"Fed-2SLS-CF quadratic ({crossfit} folds)",
                              protocol_class(basis="quadratic", robust=True, crossfit=crossfit).run(sites),
                              theta2=True))
        thetas, covs = zip(*[quadratic_2sls([r]) for r in raw(sites)])
        mt, mc = multivariate_meta(thetas, covs)
        rows.append(EstimatorRow(estimator="site quadratic 2SLS meta-analysis", theta1=mt[0], se1=np.sqrt(mc[0, 0]),
                                 theta2=mt[1], se2=np.sqrt(mc[1, 1]),
                                 abs_diff_from_pooled=max(abs(mt[0] - theta[0]), abs(mt[1] - theta[1])),
                                 what_leaves_site="two coefficients and a 2x2 covariance per site"))

    manifest_sites = manifest["sites"]
    weighted_slope = np.sum([site["n"] * site["avg_slope"] for site in manifest_sites])
    total_n = np.sum([site["n"] for site in manifest_sites])
    truth = {"theta1": manifest["theta1"], "theta2": manifest.get("theta2"),
             "avg_slope": float(weighted_slope / total_n)}
    df = pl.DataFrame([asdict(r) for r in rows])
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"fed2sls.{shape}.csv"
    df.write_csv(out)
    (out_dir / f"fed2sls.{shape}.truth.json").write_text(json.dumps(truth, indent=1))

    print(f"\n{shape}: {len(sites)} sites, N = {res.N:,}, protocol = {protocol_name}; "
          f"true theta1 = {truth['theta1']}, theta2 = {truth['theta2']}, average slope = {truth['avg_slope']:.3f}")
    with pl.Config(tbl_rows=-1, tbl_cols=-1, float_precision=4, tbl_width_chars=200, fmt_str_lengths=40):
        print(df.drop("what_leaves_site"))
    print(f"identity check |Fed-2SLS - concatenated| = {abs(res['X'] - pooled):.2e}"
          + (f", quadratic {d:.2e}" if curved and not shared else ""))
    print(f"wrote {out}")
    return df


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--shape", action="append", choices=SHAPES + ("linear_shared",),
                   help="repeatable; default quadratic")
    p.add_argument("--all", action="store_true", help="every continuous set under simulated_data/federated/")
    p.add_argument("--crossfit", type=int, default=5, help="folds for Fed-2SLS-CF")
    p.add_argument("--out", type=Path, default=Path("results"))
    a = p.parse_args()
    if a.all:
        shapes = [d.name for d in sorted(FED.iterdir())
                  if (d / "manifest.json").exists() and d.name.startswith(SHAPES)]
    else:
        shapes = a.shape or ["quadratic"]
    for shape in shapes:
        analyse(shape, a.crossfit, a.out)


if __name__ == "__main__":
    main()
