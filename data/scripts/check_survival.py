"""Fit Cox models to a simulated survival dataset and compare with the true log hazard ratio.

naive:  Cox of (time, event) on X                       confounded by U
2SPS:   Cox on X_hat, the SNP-predicted X               MR two-stage predictor substitution
2SRI:   Cox on X and the first-stage residual X - X_hat  MR two-stage residual inclusion
oracle: Cox on X and U                                  what you would get if U were measured
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from lifelines import CoxPHFitter


def fits(df: pl.DataFrame) -> dict[str, float]:
    G = df.select(pl.col("^snp.*$")).to_numpy().astype(float)
    X = df["X"].to_numpy()
    Z = np.column_stack([np.ones(len(X)), G])
    Xhat = Z @ np.linalg.lstsq(Z, X, rcond=None)[0]
    base = pd.DataFrame({"time": df["time"].to_numpy(), "event": df["event"].to_numpy(),
                         "X": X, "Xhat": Xhat, "resid": X - Xhat, "U": df["U"].to_numpy()})
    out = {}
    for name, cols in (("naive", ["X"]), ("2SPS", ["Xhat"]), ("2SRI", ["X", "resid"]), ("oracle", ["X", "U"])):
        cph = CoxPHFitter().fit(base[["time", "event", *cols]], "time", "event")
        out[name] = float(cph.params_[cols[0]])
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("dataset", type=Path, help="path without extension, e.g. simulated_data/cox")
    a = p.parse_args()
    df = pl.read_csv(a.dataset.with_suffix(".csv"))
    t = json.loads(a.dataset.with_suffix(".truth.json").read_text())
    est = fits(df)
    print(f"true log HR {t['theta']:.3f}  (HR {t['hazard_ratio']:.2f}), event rate {t['event_rate']:.2f}")
    for k, v in est.items():
        print(f"{k:7s} {v:.3f}  bias {v - t['theta']:+.3f}")


if __name__ == "__main__":
    main()
