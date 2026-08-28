# -*- coding: utf-8 -*-
"""
03_prepare_dataset.py - feature engineering and dataset splits

Input:  data/panel.csv
Output: data/dataset.xlsx (full feature table + label + split marker)
        data/splits.npz  (X/y train/valid/test, float32, incl. scaler params)

Missing-value handling (no fabricated data):
  - All features with coverage >= 30% enter the model
  - Missing values are filled with the training-set z-score mean, and a parallel
    "missing mask" column (1 = missing) lets the network know the value is
    unknown rather than actually the mean
  - Log transform: gdppc2000, manuf_pc2000, elec_gen_pc, mobile, broadband

Splits:
  - Random: 70% train / 20% valid / 10% test
  - Time: train <= 2010, test > 2010 (robustness check)

Usage: python 03_prepare_dataset.py
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

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")

MIN_COVERAGE = 0.30
LOG_FEATS = ["gdppc2000", "manuf_pc2000", "elec_gen_pc", "mobile", "broadband"]
EXCLUDE = ["iso3", "year", "country", "pop", "tfr",
           "gdppc", "manuf_pc", "agri_empl", "phys_pc"]  # redundant / non-feature columns


def main():
    print("== Dataset preparation ==")
    df = pd.read_csv(os.path.join(DATA, "panel.csv"), encoding="utf-8-sig")

    cov = json.load(open(os.path.join(DATA, "coverage.json"), encoding="utf-8"))
    feats = [c for c in df.columns
             if c not in EXCLUDE and c in cov and cov[c]["share"] >= MIN_COVERAGE]
    print(f"  features ({len(feats)}): {feats}")

    Xraw = df[feats].copy()
    for f in LOG_FEATS:
        if f in Xraw.columns:
            Xraw[f] = np.log1p(Xraw[f].clip(lower=0))
    masks = Xraw.isna().astype(np.float32)

    # Random split (all rows, missing values kept)
    full = pd.concat([df[["iso3", "year", "tfr"]], Xraw], axis=1).reset_index(drop=True)
    full = full.sample(frac=1.0, random_state=42).reset_index(drop=True)
    n_all = len(full)
    n_tr, n_va = int(n_all * 0.70), int(n_all * 0.20)
    tr, va, te = full.iloc[:n_tr], full.iloc[n_tr:n_tr + n_va], full.iloc[n_tr + n_va:]
    print(f"  random split: train {len(tr)} valid {len(va)} test {len(te)}")

    # Standardization (training-set statistics; missing filled with mean, flagged by mask)
    Xtr = tr[feats].values.astype(np.float64)
    mu = np.nanmean(Xtr, axis=0)
    sd = np.nanstd(Xtr, axis=0)
    sd[sd == 0] = 1.0
    Z = lambda X: np.nan_to_num((X - mu) / sd, nan=0.0).astype(np.float32)
    Xtr_z, Xva_z, Xte_z = Z(tr[feats].values), Z(va[feats].values), Z(te[feats].values)
    Mtr, Mva, Mte = masks.iloc[tr.index].values, masks.iloc[va.index].values, masks.iloc[te.index].values
    Xtr_in = np.concatenate([Xtr_z, Mtr], axis=1).astype(np.float32)
    Xva_in = np.concatenate([Xva_z, Mva], axis=1).astype(np.float32)
    Xte_in = np.concatenate([Xte_z, Mte], axis=1).astype(np.float32)
    ytr = tr["tfr"].values.astype(np.float32)
    yva = va["tfr"].values.astype(np.float32)
    yte = te["tfr"].values.astype(np.float32)

    # Excel output
    xl = full.copy()
    for i, f in enumerate(feats):
        xl[f"{f}_missing"] = masks[f].values
    xl["split"] = "train"
    xl.loc[va.index, "split"] = "valid"
    xl.loc[te.index, "split"] = "test"
    xl_path = os.path.join(DATA, "dataset.xlsx")
    xl.to_excel(xl_path, index=False)
    print(f"  dataset.xlsx -> {xl_path} ({xl.shape})")

    # Time split (robustness)
    t_full = pd.concat([df[["iso3", "year", "tfr"]], Xraw], axis=1)
    t_tr = t_full[t_full["year"] <= 2010]
    t_te = t_full[t_full["year"] > 2010]
    Xt_tr = np.concatenate([Z(t_tr[feats].values), masks.iloc[t_tr.index].values], axis=1).astype(np.float32)
    Xt_te = np.concatenate([Z(t_te[feats].values), masks.iloc[t_te.index].values], axis=1).astype(np.float32)
    print(f"  time split: train (<=2010) {len(t_tr)} test (>2010) {len(t_te)}")

    np.savez(os.path.join(DATA, "splits.npz"),
             X_tr=Xtr_in, y_tr=ytr, X_va=Xva_in, y_va=yva, X_te=Xte_in, y_te=yte,
             Xt_tr=Xt_tr, yt_tr=t_tr["tfr"].values.astype(np.float32),
             Xt_te=Xt_te, yt_te=t_te["tfr"].values.astype(np.float32),
             mu=mu.astype(np.float32), sd=sd.astype(np.float32),
             feats=np.array(feats, dtype=object),
             tr_idx=tr.index.values, va_idx=va.index.values, te_idx=te.index.values)
    cfg = {"features": feats, "log_feats": LOG_FEATS, "n_train": len(tr),
           "n_valid": len(va), "n_test": len(te), "n_time_tr": len(t_tr),
           "n_time_te": len(t_te), "n_inputs": Xtr_in.shape[1]}
    with open(os.path.join(DATA, "dataset_cfg.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=1)
    print("  split info -> data/dataset_cfg.json")
    print("== Done ==")


if __name__ == "__main__":
    main()
