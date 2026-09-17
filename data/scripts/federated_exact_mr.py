"""FedMR on the federated sets: exact federated 2SLS versus the other three families.

For each continuous set this runs, without ever concatenating rows across sites,

    FedMR             sufficient statistics summed over sites (scripts/fedmr.py), classical
                      and robust standard errors, first-stage F and partial R^2
    FedMR-CF          the same with k-fold cross-fitted X_hat (out-of-fold first stage)

and compares them with

    concatenated      one pooled 2SLS on the stacked rows (federated_summary_mr.py)
    site meta         each site's own 2SLS, combined by inverse-variance meta-analysis
    sumstats IVW      per-SNP GWAS effects, per-site IVW, meta-analysis

The headline number is |FedMR - concatenated|, which must be below 1e-10: the
federated estimator is the pooled estimator, not an approximation of it.

Writes results/fedmr.<shape>.csv (one row per estimator) that the forest and
dose-response plots pick up, and prints the table.

    uv run python scripts/federated_exact_mr.py --shape quadratic
    uv run python scripts/federated_exact_mr.py --all
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
import fedmr  # noqa: E402
from federated_nonlinear_mr import sumstats_slope  # noqa: E402
from federated_summary_mr import multivariate_meta, pooled_2sls, quadratic_2sls  # noqa: E402

FED = Path("simulated_data/federated")
SHAPES = ("linear", "quadratic", "ushape", "threshold")


def raw(sites):
    return [(s.G, s.X, s.Y) for s in sites]


def site_meta_linear(sites):
    est, se = zip(*[pooled_2sls([r])[:2] for r in raw(sites)])
    w = 1 / np.array(se) ** 2
    return float(np.sum(w * est) / np.sum(w)), float(np.sqrt(1 / np.sum(w)))


def row(name, theta1, se1, rse1=None, theta2=None, se2=None, rse2=None, F=None, r2=None, diff=None, leaves=""):
    return {"estimator": name, "theta1": theta1, "se1": se1, "robust_se1": rse1,
            "theta2": theta2, "se2": se2, "robust_se2": rse2,
            "first_stage_F": F, "partial_r2": r2, "abs_diff_from_pooled": diff, "what_leaves_site": leaves}


def analyse(shape, instruments, crossfit, out_dir):
    folder = FED / shape
    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest.get("shared_snps"):
        instruments = "shared"
    sites = fedmr.load_sites(folder)
    curved = shape != "linear"
    rows = []

    pooled, pooled_se, naive = pooled_2sls(raw(sites))
    rows.append(row("concatenated 2SLS", pooled, pooled_se, leaves="individual rows"))

    res = fedmr.fedmr(sites, instruments=instruments, basis="linear", robust=True)
    fs = res.first_stage["X"]
    rows.append(row("FedMR", res["X"], res.se("X"), res.se("X", True), F=fs["F"], r2=fs["partial_r2"],
                    diff=abs(res["X"] - pooled), leaves="Z'Z, Z'W, Z'Y, W'W, W'Y, Y'Y, n (+ H for robust SE)"))

    cf = fedmr.fedmr(sites, instruments=instruments, basis="linear", robust=True, crossfit=crossfit)
    rows.append(row(f"FedMR-CF ({crossfit} folds)", cf["X"], cf.se("X"), cf.se("X", True),
                    F=cf.first_stage["X"]["F"], r2=cf.first_stage["X"]["partial_r2"],
                    leaves="per-fold first-stage moments, then the same statistics"))

    meta, meta_se = site_meta_linear(sites)
    rows.append(row("site 2SLS meta-analysis", meta, meta_se, diff=abs(meta - pooled),
                    leaves="one estimate and SE per site"))
    slope, slope_se = sumstats_slope(raw(sites))
    rows.append(row("per-SNP sumstats IVW", slope, slope_se, diff=abs(slope - pooled),
                    leaves="per-SNP beta_x, beta_y, SEs"))
    rows.append(row("naive OLS (no instruments)", naive, None))

    if curved:
        theta, cov = quadratic_2sls(raw(sites))
        rows.append(row("concatenated quadratic 2SLS", theta[0], np.sqrt(cov[0, 0]), theta2=theta[1],
                        se2=np.sqrt(cov[1, 1]), leaves="individual rows"))
        q = fedmr.fedmr(sites, instruments=instruments, basis="quadratic", robust=True)
        d = max(abs(q["X"] - theta[0]), abs(q["X2"] - theta[1]))
        rows.append(row("FedMR quadratic", q["X"], q.se("X"), q.se("X", True), q["X2"], q.se("X2"), q.se("X2", True),
                        F=q.first_stage["X"]["F"], r2=q.first_stage["X"]["partial_r2"], diff=d,
                        leaves="the same statistics with W = [X, X^2, site], Z = [xhat, xhat^2, site]"))
        qcf = fedmr.fedmr(sites, instruments=instruments, basis="quadratic", robust=True, crossfit=crossfit)
        rows.append(row(f"FedMR-CF quadratic ({crossfit} folds)", qcf["X"], qcf.se("X"), qcf.se("X", True),
                        qcf["X2"], qcf.se("X2"), qcf.se("X2", True)))
        thetas, covs = zip(*[quadratic_2sls([r]) for r in raw(sites)])
        mt, mc = multivariate_meta(thetas, covs)
        rows.append(row("site quadratic 2SLS meta-analysis", mt[0], np.sqrt(mc[0, 0]), theta2=mt[1],
                        se2=np.sqrt(mc[1, 1]), diff=max(abs(mt[0] - theta[0]), abs(mt[1] - theta[1])),
                        leaves="two coefficients and a 2x2 covariance per site"))

    truth = {"theta1": manifest["theta1"], "theta2": manifest.get("theta2"),
             "avg_slope": float(np.sum([s["n"] * s["avg_slope"] for s in manifest["sites"]])
                              / np.sum([s["n"] for s in manifest["sites"]]))}
    df = pl.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"fedmr.{shape}.csv"
    df.write_csv(out)
    (out_dir / f"fedmr.{shape}.truth.json").write_text(json.dumps(truth, indent=1))

    print(f"\n{shape}: {len(sites)} sites, N = {res.N:,}, instruments = {instruments}; "
          f"true theta1 = {truth['theta1']}, theta2 = {truth['theta2']}, average slope = {truth['avg_slope']:.3f}")
    with pl.Config(tbl_rows=-1, tbl_cols=-1, float_precision=4, tbl_width_chars=200, fmt_str_lengths=40):
        print(df.drop("what_leaves_site"))
    print(f"identity check |FedMR - concatenated| = {abs(res['X'] - pooled):.2e}"
          + (f", quadratic {d:.2e}" if curved else ""))
    print(f"wrote {out}")
    return df


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--shape", action="append", choices=SHAPES + ("linear_shared",),
                   help="repeatable; default quadratic")
    p.add_argument("--all", action="store_true", help="every continuous set under simulated_data/federated/")
    p.add_argument("--instruments", choices=["site", "shared"], default="site",
                   help="site: each site's SNPs are its own variants (block-diagonal Z'Z); "
                        "shared: harmonized SNPs, one column per SNP. Sets whose manifest says "
                        "shared_snps use 'shared' regardless.")
    p.add_argument("--crossfit", type=int, default=5, help="folds for FedMR-CF")
    p.add_argument("--out", type=Path, default=Path("results"))
    a = p.parse_args()
    if a.all:
        shapes = [d.name for d in sorted(FED.iterdir())
                  if (d / "manifest.json").exists() and d.name.startswith(SHAPES)]
    else:
        shapes = a.shape or ["quadratic"]
    for shape in shapes:
        analyse(shape, a.instruments, a.crossfit, a.out)


if __name__ == "__main__":
    main()
