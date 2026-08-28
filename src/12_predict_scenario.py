# -*- coding: utf-8 -*-
"""
12_predict_scenario.py - scenario forecasting: the user sets one or more parameters at
a future year and the script predicts the fertility rate in that year

Rules:
  - Baseline = TFR and all feature values of the latest available year L for the country
    (2024 for all 83 countries)
  - The user sets the values of several features at target year T (original units);
    unset features keep their year-L values
  - The model predicts the 5-year dTFR, chained forward to T (a final partial window
    shorter than 5 years is scaled linearly, an approximation)
  - Output: TFR(T), the reduction relative to the latest year, and a per-parameter
    reduction factor analysis (single-effect counterfactuals)
  - Factor analysis: contribution of a feature = dTFR under all settings minus the
    dTFR when that feature keeps its year-L value

Usage:
  python 12_predict_scenario.py --iso3 CHN --year 2035 --set '{"edu_f": 13.0}'
  python 12_predict_scenario.py --iso3 CHN --year 2035 --set '{"edu_f": 13.0, "mat_leave_days": 180.0}'
  python 12_predict_scenario.py --iso3 KOR --year 2050 --set '{"gdppc2000": 30000.0, "manuf_pc2000": 8000.0}'
  python 12_predict_scenario.py --iso3 CHN --year 2035 --set '{"edu_f": 13.0}' --set '{"literacy": 99.0}'
  python 12_predict_scenario.py --iso3 NGA --year 2045 --reach 2040 --set '{"elec_gen_pc": 538.5, "edu_f": 5.26}'
      # --reach: set features reach the given values in 2040 and stay there; forecast 2045
  python 12_predict_scenario.py --list-countries   # list all available countries

Country resolution (--iso3, case-insensitive, tried in order):
  1) ISO3 code:      CHN, chn, nGA  -> the code itself
  2) Canonical name: Nigeria, nigeria, China, VIET NAM, Viet Nam
  3) Common aliases: usa/us/america->USA, uk/britain->GBR, korea->KOR,
                     taiwan->TWN, turkey/turkiye->TUR, vietnam/"viet nam"->VNM
  4) Unique substring: "viet"->VNM, "congo"->COD, "syria"->SYR; an error with a hint
                     is raised when multiple countries match
  Example: --iso3 "viet nam"  ==  --iso3 VNM  ==  --iso3 vietnam
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

from nn_common import (ALL_FEATS, DATA, build_input, country_list, g, load_model,
                       load_panel, predict, resolve_country)

K = 5


def g_at_year(y, L, R, gL, gT, set_feats):
    """Transform-space value of each feature at year y: set features interpolate
    linearly over [L, R] and stay constant after R; unset features keep gL"""
    out = {}
    for f in gL:
        if f not in set_feats:
            out[f] = gL[f]
        elif np.isnan(gL[f]):
            out[f] = gT[f]
        elif np.isnan(gT[f]):
            out[f] = gL[f]
        elif y <= R:
            out[f] = gL[f] + (gT[f] - gL[f]) * (y - L) / (R - L)
        else:
            out[f] = gT[f]
    return out


def chain_predict(L, T, tfrL, gL, gT, set_feats, layers, mu, sd, R=None):
    """Chained rolling prediction of TFR(T); returns (TFR(T), [(year, TFR, dTFR), ...])"""
    R = T if R is None else R
    n_full = (T - L) // K
    grid = [L + K * i for i in range(n_full + 1)]
    if grid[-1] != T:
        grid.append(T)
    tfr, steps = tfrL, []
    for a, b in zip(grid[:-1], grid[1:]):
        ga, gb = g_at_year(a, L, R, gL, gT, set_feats), g_at_year(b, L, R, gL, gT, set_feats)
        d = float(predict(build_input(tfr, ga, gb, ALL_FEATS, mu, sd), layers)[0])
        if b - a != K:
            d *= (b - a) / K          # final partial window: linear scaling (approximation)
        tfr += d
        steps.append((b, tfr, d))
    return tfr, steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iso3", type=str, default="CHN",
                    help="country: ISO3 code or name (case-insensitive; aliases and unique "
                         "substrings supported, e.g. 'viet nam' -> VNM)")
    ap.add_argument("--year", type=int, default=None,
                    help="target year (must be later than the latest data year 2024; "
                         "optional with --list-countries)")
    ap.add_argument("--list-countries", action="store_true",
                    help="list all available countries (code=name)")
    ap.add_argument("--reach", type=int, default=None,
                    help="year in which set features reach the given values (default = "
                         "target year, then stay constant)")
    ap.add_argument("--set", action="append", default=[], metavar="JSON",
                    help='set target-year feature values (original units), repeatable; '
                         'e.g. {"edu_f": 13.0}')
    args = ap.parse_args()

    fmt0 = lambda v: "missing" if v is None else f"{v:g}"

    z, layers, mu, sd = load_model("delta_k5")
    df = load_panel()
    if args.list_countries:
        print("  available countries (ISO3=name):\n  " + country_list(df)); return
    if args.year is None:
        print("  error: --year required to set the target year (or use --list-countries)"); return
    code, hint = resolve_country(args.iso3, df)
    if hint and code:
        print(f"  hint: {hint}")
    if not code:
        print(f"  error: {hint}\n  available countries (ISO3=name):\n  " + country_list(df)); return
    d = df[df["iso3"] == code].sort_values("year")
    rL = d[d["tfr"].notna()].iloc[-1]
    L, tfrL = int(rL["year"]), float(rL["tfr"])
    T = args.year
    if T <= L:
        print(f"  error: target year {T} must be later than the latest data year {L}"); return
    R = T if args.reach is None else args.reach
    if not (L < R <= T):
        print(f"  error: --reach {R} must satisfy {L} < reach <= target year {T}"); return

    over = {}
    for s in args.set:
        try:
            j = json.loads(s)
        except json.JSONDecodeError:
            print(f"  error: --set must be valid JSON: {s}"); return
        if not isinstance(j, dict):
            print(f"  error: --set must be a JSON object: {s}"); return
        over.update(j)
    if not over:
        print("  error: set at least one parameter with --set, e.g. --set '{\"edu_f\": 13.0}'"); return
    bad = [f for f in over if f not in ALL_FEATS]
    if bad:
        print(f"  error: unknown parameters {bad}. Available: {ALL_FEATS}"); return

    gL = {f: (g(f, float(rL[f])) if np.isfinite(rL[f]) else np.nan) for f in ALL_FEATS}
    rawL = {f: (float(rL[f]) if np.isfinite(rL[f]) else None) for f in ALL_FEATS}
    gT = {f: g(f, float(over[f])) for f in ALL_FEATS if f in over}
    for f in ALL_FEATS:
        gT.setdefault(f, gL[f])
    set_feats = set(over)
    for f in over:
        if rawL[f] is not None and float(over[f]) < 0 and f in ("gdppc2000", "manuf_pc2000",
                                                                "elec_gen_pc", "mobile", "broadband"):
            print(f"  warning: {f} is a log feature; the negative value {over[f]} will not be log-transformed")

    tfrT, steps = chain_predict(L, T, tfrL, gL, gT, set_feats, layers, mu, sd, R)
    red = tfrL - tfrT

    print(f"\n[Scenario forecast] {rL['country']}({args.iso3}) {L} -> {T}")
    print(f"  baseline: TFR({L}) = {tfrL:.3f} (latest year with data, WPP)")
    s_vals = ", ".join(f"{f} {fmt0(rawL[f])}->{fmt0(float(over[f]))}" for f in over)
    print(f"  set (original units): {s_vals}")
    print(f"  {len(ALL_FEATS) - len(set_feats)} unset parameters keep their year-{L} values")
    miss = [f for f in ALL_FEATS if rawL[f] is None]
    if miss:
        print(f"  note: the following parameters already lack data in {L}; "
              f"'keeping' means the model treats them as unknown: {miss}")
    chain = " | ".join(f"{y}:{t:.3f}({'down' if tfrL - t >= 0 else 'up'}{abs(tfrL - t):.3f})"
                       for y, t, _ in steps)
    print(f"  chained path: {chain}")
    if (T - L) % K:
        print(f"  (note: final {(T - L) % K} years scaled linearly from the 5-year prediction, approximation)")
    print(f"\n  result: TFR({T}) = {tfrT:.3f}")
    if red >= 0:
        print(f"  reduction: -{red:.3f} relative to {L} ({red / tfrL:.1%})")
    else:
        print(f"  change: +{-red:.3f} (an increase relative to {L})")

    print("\n  Reduction factor analysis (marginal contributions vs the zero-change baseline):")
    fmt = lambda v: "missing" if v is None else f"{v:g}"
    total_d = tfrT - tfrL
    tfr_base, _ = chain_predict(L, T, tfrL, gL, gT, set(), layers, mu, sd, R)
    d_base = tfr_base - tfrL
    print(f"    zero-change baseline (all parameters at year-{L} values): dTFR {d_base:+.3f} "
          f"(TFR({T}) = {tfr_base:.3f})")
    contribs = []
    for f in sorted(set_feats):
        gT_cf = dict(gT)
        gT_cf[f] = gL[f]
        tfr_cf, _ = chain_predict(L, T, tfrL, gL, gT_cf, set_feats, layers, mu, sd, R)
        c = total_d - (tfr_cf - tfrL)
        contribs.append((f, c))
        mark = "" if rawL[f] is not None else " (this parameter is missing in the latest year; unset = model treats as unknown)"
        print(f"    {f:16s} {fmt(rawL[f])}->{fmt(float(over[f]))}: dTFR contribution {c:+.3f} "
              f"(unset -> TFR({T}) = {tfr_cf:.3f}){mark}")
    inter = total_d - d_base - sum(c for _, c in contribs)
    if len(contribs) > 1:
        print(f"    interaction residual (sum of marginal effects != total effect): {inter:+.3f}")
    if red > 0:
        items = [("zero-change baseline", d_base)] + contribs + [("interaction residual", inter)]
        parts = []
        for n, v in items:
            if v <= 0:
                parts.append(f"{n} {-100 * v / red:.1f}%")
            else:
                parts.append(f"{n} offsets {100 * v / red:.1f}%")
        print("    composition of the reduction (" + f"{red:.3f}" + "): " + " | ".join(parts))

    imp_path = os.path.join(DATA, "feature_selection_delta_k5.json")
    if os.path.exists(imp_path):
        imp = json.load(open(imp_path, encoding="utf-8"))["importance"]
        ref = " | ".join(f"{f} {imp[f]:.3f}" for f, _ in contribs)
        print(f"  reference (global permutation importance dR2): {ref}")


if __name__ == "__main__":
    main()
