# -*- coding: utf-8 -*-
"""
10_formula_delta.py - compact mathematical expression of dTFR (linear increment equation)

    dTFR(t -> t+k) = b0 + b_tfr * TFR(t) + sum(b_i * X_i(t)) + sum(g_i * dX_i(t -> t+k))

Fit: time-train windows (complete cases <= 2010), evaluation: time test (>= 2011)
Output: data/formula_delta.json (with original-scale coefficients)
Usage: python 10_formula_delta.py [--k 5]
"""
import argparse
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

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")

LOG_FEATS = ["gdppc2000", "manuf_pc2000", "elec_gen_pc", "mobile", "broadband"]
BASE_FEATS = ["urban", "elec_access", "mobile", "broadband", "literacy",
              "physicians", "elec_gen_pc", "edu_f", "mat_leave_days",
              "pat_leave_days", "nonagri_empl", "gdppc2000", "manuf_pc2000"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--sel", action="store_true", help="use only the feature-selection minimal set")
    args = ap.parse_args()
    k = args.k

    tag = f"delta_k{k}"
    feats = BASE_FEATS + ["rail_pc"]
    if args.sel:
        sel = json.load(open(os.path.join(DATA, f"feature_selection_{tag}.json"), encoding="utf-8"))
        feats = [f for f in feats if f in sel["selected"]]
    print(f"  features: {feats}, k={k}{' (minimal set)' if args.sel else ''}")

    df = pd.read_csv(os.path.join(DATA, "panel.csv"), encoding="utf-8-sig")
    df["rail_pc"] = df["rail_km"] / (df["pop"] * 1000.0)
    df = df.sort_values(["iso3", "year"])

    def g(f, v):
        if f in LOG_FEATS and pd.notna(v) and v >= 0:
            return np.log1p(v)
        return v

    rows = []
    for iso, grp in df.groupby("iso3"):
        grp = grp.set_index("year")
        for t in range(1950, 2025 - k):
            if t not in grp.index or t + k not in grp.index:
                continue
            r0, rk = grp.loc[t], grp.loc[t + k]
            if pd.isna(r0["tfr"]) or pd.isna(rk["tfr"]):
                continue
            row = {"iso3": iso, "year": t, "year_end": t + k,
                   "dTFR": rk["tfr"] - r0["tfr"], "tfr0": r0["tfr"]}
            for f in feats:
                v0, vk = g(f, r0[f]), g(f, rk[f])
                row[f] = v0
                row[f"{f}_d"] = vk - v0 if (pd.notna(v0) and pd.notna(vk)) else np.nan
            rows.append(row)
    d = pd.DataFrame(rows)

    cols = ["tfr0"] + feats + [f"{f}_d" for f in feats]
    # Consistent with the NN: use all windows with mean-fill (missing rows kept),
    # coefficients in standardized units
    z = np.load(os.path.join(DATA, f"splits_{tag}.npz"), allow_pickle=True)
    Xt_tr, yt_tr, Xt_te, yt_te = z["Xt_tr"], z["yt_tr"], z["Xt_te"], z["yt_te"]
    meta = z["meta"]
    n_f = len(cols)
    Xtr = Xt_tr[:, :n_f].astype(np.float64)
    Xte = Xt_te[:, :n_f].astype(np.float64)
    ytr, yte = yt_tr.astype(np.float64), yt_te.astype(np.float64)
    print(f"  mean-filled samples: train {len(Xtr)} test {len(Xte)} (standardized units)")
    tfr0_te = np.array([float(r[3]) for r in meta])
    mask_te = np.array([int(r[1]) for r in meta]) >= 2011

    Atr = np.concatenate([np.ones((len(Xtr), 1)), Xtr], axis=1)
    beta, *_ = np.linalg.lstsq(Atr, ytr, rcond=None)
    yp_tr, yp_te = Atr @ beta, np.concatenate([np.ones((len(Xte), 1)), Xte], axis=1) @ beta
    r2f = lambda a, b: 1 - np.sum((a - b) ** 2) / np.sum((a - a.mean()) ** 2)
    mae_f = lambda a, b: np.mean(np.abs(a - b))
    succ_f = lambda a, b: np.mean(np.abs(a - b) <= 0.25)

    print("\n== dTFR linear equation (original scale) ==")
    print(f"  dTFR({k}y) = {beta[0]:+.4f}")
    for c, b in zip(cols, beta[1:]):
        print(f"      {b:+.4f} x {c}")
    print(f"\n  train (<=2010): R2={r2f(ytr, yp_tr):.4f} MAE={mae_f(ytr, yp_tr):.4f}")
    print(f"  test (>=2011): R2={r2f(yte, yp_te):.4f} MAE={mae_f(yte, yp_te):.4f} "
          f"|err|<=0.25={succ_f(yte, yp_te):.2%}")
    tfr_t = tfr0_te[mask_te] + yp_te
    tfr_a = tfr0_te[mask_te] + yte
    print(f"  TFR level (test): MAE={mae_f(tfr_a, tfr_t):.4f} R2={r2f(tfr_a, tfr_t):.4f} "
          f"|err|<=0.25={succ_f(tfr_a, tfr_t):.2%}")

    out = {"k": k, "features": cols, "beta": beta.tolist(),
           "train_r2": float(r2f(ytr, yp_tr)), "test_r2": float(r2f(yte, yp_te)),
           "test_mae": float(mae_f(yte, yp_te)), "test_succ25": float(succ_f(yte, yp_te)),
           "tfr_test_mae": float(mae_f(tfr_a, tfr_t)), "tfr_test_r2": float(r2f(tfr_a, tfr_t)),
           "tfr_test_succ25": float(succ_f(tfr_a, tfr_t)),
           "n_train": len(Xtr), "n_test": len(Xte)}
    with open(os.path.join(DATA, f"formula_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"  -> data/formula_{tag}.json")


if __name__ == "__main__":
    main()
