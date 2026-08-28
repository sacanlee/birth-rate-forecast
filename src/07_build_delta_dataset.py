# -*- coding: utf-8 -*-
"""
07_build_delta_dataset.py - restructure training data: predict delta-TFR rather than
the absolute level (user follow-up requirement 2)

Design (accounting for lagged effects):
  Target: dTFR(t -> t+k) = TFR(t+k) - TFR(t),  k = 5 years (default, adjustable)
  Input:  [TFR(t) current level, X(t) feature levels (14), dX(t -> t+k) feature changes (14),
           masks (28)]
          - current levels capture lagged effects (V5 convergence-term idea),
            changes capture incremental effects
  Features (14): 13 original features + rail_pc (rail km per capita, WDI/UIC)
  Masks: level-missing mask 14 + change-missing mask 14 (dX masked if either endpoint missing)
  Splits: random 70/20/10 + time (train t+k <= 2010, test t >= 2011, no overlap)

Output: data/delta_k5.npz / data/dataset_delta_k5.xlsx
Usage: python 07_build_delta_dataset.py [--k 5]
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
# rail_pc is computed from the interpolated rail_km
BASE_FEATS = ["urban", "elec_access", "mobile", "broadband", "literacy",
              "physicians", "elec_gen_pc", "edu_f", "mat_leave_days",
              "pat_leave_days", "nonagri_empl", "gdppc2000", "manuf_pc2000"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()
    k = args.k

    df = pd.read_csv(os.path.join(DATA, "panel.csv"), encoding="utf-8-sig")
    df = df.sort_values(["iso3", "year"])
    # Rail per capita (from the interpolated rail_km)
    df["rail_pc"] = df["rail_km"] / (df["pop"] * 1000.0)
    feats = BASE_FEATS + ["rail_pc"]
    print(f"  features ({len(feats)}): {feats},  horizon k={k}")

    def g(f, v):
        """Log transform (skewed features): both level and change use log1p differences"""
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
    print(f"  dTFR window samples: {len(d)}  dTFR range [{d['dTFR'].min():.2f}, {d['dTFR'].max():.2f}]")

    level_feats = feats
    delta_feats = [f"{f}_d" for f in feats]
    all_feats = ["tfr0"] + level_feats + delta_feats

    # Standardization (training-set statistics) + masks
    d = d.sample(frac=1.0, random_state=42).reset_index(drop=True)
    Xraw = d[all_feats].astype(np.float64)
    masks = Xraw.isna().astype(np.float32)
    tr_n = int(len(d) * 0.70)
    va_n = int(len(d) * 0.20)
    mu = Xraw.iloc[:tr_n].mean().values
    sd = Xraw.iloc[:tr_n].std().values.copy()
    sd[sd == 0] = 1.0
    Z = lambda X: np.nan_to_num((X - mu) / sd, nan=0.0).astype(np.float32)
    Xz = Z(Xraw.values)
    X = np.concatenate([Xz, masks.values], axis=1).astype(np.float32)
    y = d["dTFR"].values.astype(np.float32)

    tr, va, te = X[:tr_n], X[tr_n:tr_n + va_n], X[tr_n + va_n:]
    ytr, yva, yte = y[:tr_n], y[tr_n:tr_n + va_n], y[tr_n + va_n:]
    print(f"  random split: train {len(tr)} valid {len(va)} test {len(te)}")

    # Time split (no overlap): train t+k <= 2010
    t_tr_mask = d["year_end"] <= 2010
    t_te_mask = d["year"] >= 2011
    Xt_tr, yt_tr = X[t_tr_mask.values], y[t_tr_mask.values]
    Xt_te, yt_te = X[t_te_mask.values], y[t_te_mask.values]
    print(f"  time split: train (window <=2010) {len(Xt_tr)} test (window >=2011) {len(Xt_te)}")

    tag = f"delta_k{k}"
    np.savez(os.path.join(DATA, f"splits_{tag}.npz"),
             X_tr=tr, y_tr=ytr, X_va=va, y_va=yva, X_te=te, y_te=yte,
             Xt_tr=Xt_tr, yt_tr=yt_tr, Xt_te=Xt_te, yt_te=yt_te,
             mu=mu.astype(np.float32), sd=sd.astype(np.float32),
             feats=np.array(all_feats, dtype=object),
             tr_idx=d.index[:tr_n].values, va_idx=d.index[tr_n:tr_n + va_n].values,
             te_idx=d.index[tr_n + va_n:].values,
             meta=np.array(d[["iso3", "year", "year_end", "tfr0"]].astype(str).values, dtype=object))
    d.to_excel(os.path.join(DATA, f"dataset_{tag}.xlsx"), index=False)
    cfg = {"k": k, "features": all_feats, "n": len(d), "n_train": len(tr),
           "n_valid": len(va), "n_test": len(te), "n_time_tr": len(Xt_tr), "n_time_te": len(Xt_te),
           "n_inputs": X.shape[1], "dTFR_min": float(d["dTFR"].min()), "dTFR_max": float(d["dTFR"].max())}
    with open(os.path.join(DATA, f"cfg_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=1)
    print(f"  -> data/splits_{tag}.npz, dataset_{tag}.xlsx, cfg_{tag}.json (input dim {X.shape[1]})")


if __name__ == "__main__":
    main()
