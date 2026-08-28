# -*- coding: utf-8 -*-
"""
11_predict_country.py - predict the fertility decline of a country over the next k
years with the dTFR model + re-run of all model performance metrics

Usage:
  Re-run all metrics:            python 11_predict_country.py --eval
  Verify a known window (2019->2024 China): python 11_predict_country.py --iso3 CHN --year 2019
  Forward prediction (out of sample, dX extrapolated by recent 5-year trend):
                                 python 11_predict_country.py --iso3 CHN --year 2024
  Scenario (set specific t+5 feature values, rest by trend):
                                 python 11_predict_country.py --iso3 CHN --year 2024 \
                                     --scenario '{"gdppc2000": 13000.0, "mat_leave_days": 158.0}'
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd

from nn_common import (ALL_FEATS, DATA, build_input, country_list, g, load_model,
                       load_panel, mae, predict, r2, resolve_country, succ)


def past_get(d, year, f):
    r = d[d["year"] == year]
    return np.nan if r.empty or pd.isna(r.iloc[0][f]) else float(r.iloc[0][f])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--eval", action="store_true", help="re-run all performance metrics")
    ap.add_argument("--iso3", type=str, default="CHN",
                    help="country: ISO3 code or name (case-insensitive; aliases and unique "
                         "substrings supported, e.g. 'viet nam' -> VNM)")
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--scenario", type=str, default=None,
                    help='JSON setting t+5 feature values, e.g. {"gdppc2000": 13000.0}; '
                         'unspecified ones use the recent 5-year actual change')
    args = ap.parse_args()
    k = args.k
    tag = f"delta_k{k}"
    z, layers, mu, sd = load_model(tag)
    feats = ALL_FEATS
    n_params = sum(W.size + b.size for W, b in layers)

    if args.eval:
        for name, X, y in [("train", z["X_tr"], z["y_tr"]), ("valid", z["X_va"], z["y_va"]),
                           ("test (random)", z["X_te"], z["y_te"]), ("test (time)", z["Xt_te"], z["yt_te"])]:
            yp = predict(X, layers)
            print(f"  [{name}] dTFR MAE {mae(y, yp):.4f} R2 {r2(y, yp):.4f} "
                  f"|err|<=0.25 {succ(y, yp, 0.25):.2%} |err|<=0.50 {succ(y, yp, 0.50):.2%}")
        tfr0_te = np.array([float(r[3]) for r in z["meta"]])
        mask_te = np.array([int(r[1]) for r in z["meta"]]) >= 2011
        yt_te = z["yt_te"]
        tfr_p, tfr_t = tfr0_te[mask_te] + predict(z["Xt_te"], layers), tfr0_te[mask_te] + yt_te
        print(f"  [test (time)] TFR level MAE {mae(tfr_t, tfr_p):.4f} R2 {r2(tfr_t, tfr_p):.4f} "
              f"|err|<=0.25 {succ(tfr_t, tfr_p, 0.25):.2%}")
        print(f"  parameters: {n_params:,}  baselines: d=0 persistence MAE~0.35, linear MAE 0.265")
        return

    df = load_panel()
    code, hint = resolve_country(args.iso3, df)
    if hint and code:
        print(f"  hint: {hint}")
    if not code:
        print(f"  error: {hint}\n  available countries (ISO3=name):\n  " + country_list(df)); return
    d = df[df["iso3"] == code].sort_values("year")
    t0 = d[d["year"] == args.year]
    if t0.empty:
        print(f"  no {args.iso3} data for {args.year} in the panel"); return
    r0 = t0.iloc[0]
    tfr0 = float(r0["tfr"])
    g0 = {f: (g(f, float(r0[f])) if pd.notna(r0[f]) else np.nan) for f in feats}
    t5 = d[d["year"] == args.year + k]
    if not t5.empty:
        r5 = t5.iloc[0]
        g5 = {f: (g(f, float(r5[f])) if pd.notna(r5[f]) else np.nan) for f in feats}
        src = "actual data"
    else:
        g5 = {}
        y1 = int(d[d["year"] < args.year]["year"].max())
        for f in feats:
            v1, vp = past_get(d, y1, f), past_get(d, y1 - k, f)
            if pd.notna(v1) and pd.notna(vp):
                g5[f] = g0[f] + (g(f, v1) - g(f, vp))   # last-5-year slope extrapolation (transform space)
            else:
                g5[f] = g0[f]
        src = f"recent 5-year trend extrapolation (based on {y1-k}->{y1}, transform space)"
    if args.scenario:
        over = json.loads(args.scenario)
        for f, v in over.items():
            if f in feats:
                g5[f] = g(f, float(v))               # scenario values given in original units
        src += f" + scenario overrides {list(over)}"
    X = build_input(tfr0, g0, g5, feats, mu, sd)
    dTFR = float(predict(X, layers)[0])
    tfr_pred = tfr0 + dTFR
    print(f"  {args.iso3} {args.year}->{args.year + k}: TFR {tfr0:.3f} -> predicted {tfr_pred:.3f} "
          f"(dTFR {dTFR:+.3f}, based on {src})")
    if not t5.empty:
        print(f"  actual: {float(r5['tfr']):.3f}  error {tfr_pred - float(r5['tfr']):+.3f}")
    return tfr_pred


if __name__ == "__main__":
    main()
