# -*- coding: utf-8 -*-
"""
06_extract_formula.py - a compact mathematical expression on the minimal feature set

Form (generalized logistic):
    TFR = Tmin + (Tmax - Tmin) / (1 + exp(beta0 + sum(beta_i * x_i)))
x_i are the selected features (log-transformed + standardized); coefficients are
reconverted to original-scale output.

Output: docs/formula.json + printed formula and goodness-of-fit (test set)
Usage: python 06_extract_formula.py
"""
import json
import os
import sys

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")

LOG_FEATS = ["gdppc2000", "manuf_pc2000", "elec_gen_pc", "mobile", "broadband"]


def main():
    sel = json.load(open(os.path.join(DATA, "feature_selection.json"), encoding="utf-8"))
    feats = sel["selected"]
    print(f"  selected minimal feature set: {feats}")

    df = pd.read_csv(os.path.join(DATA, "panel.csv"), encoding="utf-8-sig")
    cfg = json.load(open(os.path.join(DATA, "dataset_cfg.json"), encoding="utf-8"))
    # Time-extrapolation evaluation: train <= 2010 / test > 2010 (consistent with 03)
    d = df[["iso3", "year", "tfr"] + feats].copy()
    for f in LOG_FEATS:
        if f in d.columns:
            d[f] = np.log1p(d[f].clip(lower=0))
    tr = d[d["year"] <= 2010].dropna()
    te = d[d["year"] > 2010].dropna()
    print(f"  fit sample: {len(tr)} (<=2010), evaluation: {len(te)} (>2010)")

    Xtr = tr[feats].values.astype(np.float64)
    ytr = tr["tfr"].values.astype(np.float64)
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd == 0] = 1.0
    Xtr_z = (Xtr - mu) / sd
    Xte_z = (te[feats].values - mu) / sd
    yte = te["tfr"].values.astype(np.float64)

    def model(theta, X):
        tmin, tmax, b0, *bs = theta
        z = b0 + X @ np.array(bs)
        return tmin + (tmax - tmin) / (1 + np.exp(-z))

    def resid(theta, X, y):
        return model(theta, X) - y

    theta0 = np.array([1.0, 7.5, 0.5] + [0.0] * len(feats))
    res = least_squares(resid, theta0, args=(Xtr_z, ytr), max_nfev=5000)
    th = res.x
    tmin, tmax, b0 = th[0], th[1], th[2]
    bs = th[3:]

    # Reconvert to original-scale coefficients (beta_orig = beta_std / sd)
    bs_orig = bs / sd
    b0_orig = b0 - np.sum(bs_orig * mu)

    yp_tr = model(th, Xtr_z)
    yp_te = model(th, Xte_z)
    r2t = lambda y1, y2: 1 - np.sum((y1 - y2) ** 2) / np.sum((y1 - y1.mean()) ** 2)
    mae = lambda y1, y2: np.mean(np.abs(y1 - y2))
    succ = lambda y1, y2: np.mean(np.abs(y1 - y2) <= 0.25)

    print("\n== Generalized logistic formula (original scale) ==")
    print(f"  TFR = {tmin:.2f} + ({tmax:.2f} - {tmin:.2f}) / (1 + exp(-z))")
    print(f"  z = {b0_orig:+.4f}")
    for f, b in zip(feats, bs_orig):
        print(f"      {b:+.4f} x {f}")
    print(f"\n  train (<=2010): R2={r2t(ytr, yp_tr):.4f}  MAE={mae(ytr, yp_tr):.4f}")
    print(f"  test (>2010): R2={r2t(yte, yp_te):.4f}  MAE={mae(yte, yp_te):.4f}  "
          f"|err|<=0.25={succ(yte, yp_te):.2%}")

    out = {"features": feats, "Tmin": float(tmin), "Tmax": float(tmax),
           "b0_std": float(b0), "beta_std": bs.tolist(),
           "b0_raw": float(b0_orig), "beta_raw": bs_orig.tolist(),
           "mu": mu.tolist(), "sd": sd.tolist(),
           "train_r2": float(r2t(ytr, yp_tr)), "test_r2": float(r2t(yte, yp_te)),
           "test_mae": float(mae(yte, yp_te)), "test_succ25": float(succ(yte, yp_te)),
           "n_train": len(tr), "n_test": len(te)}
    with open(os.path.join(DATA, "formula.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\n  -> data/formula.json")


if __name__ == "__main__":
    main()
