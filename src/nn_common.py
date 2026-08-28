# -*- coding: utf-8 -*-
"""nn_common.py - shared utilities for the delta-TFR model (reused by scripts 11/12)

Must match the transforms in 07_build_delta_dataset.py:
  log features log1p -> z-score with training-set statistics -> missing filled with 0 + mask 1
"""
import os
import re

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
MODELS = os.path.join(BASE, "models")

LOG_FEATS = ["gdppc2000", "manuf_pc2000", "elec_gen_pc", "mobile", "broadband"]
BASE_FEATS = ["urban", "elec_access", "mobile", "broadband", "literacy",
              "physicians", "elec_gen_pc", "edu_f", "mat_leave_days",
              "pat_leave_days", "nonagri_empl", "gdppc2000", "manuf_pc2000"]
ALL_FEATS = BASE_FEATS + ["rail_pc"]


def g(f, v):
    if f in LOG_FEATS and pd.notna(v) and v >= 0:
        return np.log1p(v)
    return v


def load_model(tag="delta_k5"):
    """Return (z, layers, mu, sd); z holds splits, layers are feed-forward weights"""
    z = np.load(os.path.join(DATA, f"splits_{tag}.npz"), allow_pickle=True)
    w = np.load(os.path.join(MODELS, f"final_weights_{tag}.npz"), allow_pickle=True)
    layers = [([W, b]) for W, b in zip(w["Ws"], w["bs"])]
    return z, layers, z["mu"], z["sd"]


def predict(X, layers):
    H = X
    for i, (W, b) in enumerate(layers):
        H = H @ W + b
        if i < len(layers) - 1:
            H = np.maximum(H, 0)
    return H[:, 0]


def build_input(tfr0, g0, g5, feats, mu, sd):
    """Build the 58-dim input from transformed levels g0/g5 (tfr0 + 14 levels + 14 deltas + 29 masks)"""
    all_feats = ["tfr0"] + feats + [f"{f}_d" for f in feats]
    vals = {"tfr0": tfr0}
    for f in feats:
        v0, v5 = g0[f], g5[f]
        vals[f] = v0
        vals[f"{f}_d"] = (v5 - v0) if (pd.notna(v0) and pd.notna(v5)) else np.nan
    Xraw = np.array([vals[f] for f in all_feats], dtype=np.float64)
    mask = np.isnan(Xraw).astype(np.float32)
    Xz = np.nan_to_num((Xraw - mu) / sd, nan=0.0).astype(np.float32)
    return np.concatenate([Xz, mask])[None, :].astype(np.float32)


def mae(y1, y2):
    return float(np.mean(np.abs(y1 - y2)))


def r2(y1, y2):
    ss = np.sum((y1 - np.mean(y1)) ** 2)
    return float(1.0 - np.sum((y1 - y2) ** 2) / ss if ss > 0 else np.nan)


def succ(y1, y2, tol):
    return float(np.mean(np.abs(y1 - y2) <= tol))


def load_panel():
    df = pd.read_csv(os.path.join(DATA, "panel.csv"), encoding="utf-8-sig")
    df["rail_pc"] = df["rail_km"] / (df["pop"] * 1000.0)
    return df


def _norm_name(s):
    """Normalize a name: lowercase, non-alphanumerics -> spaces, collapse whitespace
    ('Viet Nam'/'viet-nam'/'VIETNAM' -> 'viet nam')"""
    return re.sub(r"\s+", " ", re.sub(r"[^0-9a-z]+", " ", str(s).lower())).strip()


# Common aliases (keys must be _norm_name-normalized; the panel's canonical names
# are already covered by exact/substring matching)
_COUNTRY_ALIASES = {
    "usa": "USA", "united states": "USA", "united states of america": "USA",
    "america": "USA", "us": "USA",
    "uk": "GBR", "united kingdom": "GBR", "britain": "GBR", "great britain": "GBR",
    "england": "GBR",
    "china": "CHN", "prc": "CHN", "peoples republic of china": "CHN",
    "russia": "RUS",
    "south korea": "KOR", "republic of korea": "KOR", "korea": "KOR",
    "north korea": "PRK", "dprk": "PRK", "democratic peoples republic of korea": "PRK",
    "taiwan": "TWN",
    "vietnam": "VNM", "viet nam": "VNM",
    "turkey": "TUR", "turkiye": "TUR",
    "czechia": "CZE", "czech republic": "CZE",
    "cote d ivoire": "CIV", "ivory coast": "CIV",
    "iran": "IRN",
    "myanmar": "MMR", "burma": "MMR",
    "syria": "SYR",
}


def resolve_country(query, df):
    """Resolve a user-provided country name to an ISO3 code, case-insensitive.

    Match order: 1) ISO3 code  2) canonical panel name (exact)  3) common aliases
    4) unique substring. Returns (iso3, hint) or (None, error); substring matches
    attach a hint to prevent misidentification.
    """
    q = _norm_name(query)
    if not q:
        return None, "Country input is empty"
    codes = {c.upper() for c in df["iso3"].unique()}
    if q.upper() in codes:
        return q.upper(), None
    name_map = {}
    for c, n in zip(df["iso3"], df["country"]):
        name_map.setdefault(_norm_name(n), c)
    if q in name_map:
        return name_map[q], None
    if q in _COUNTRY_ALIASES:
        return _COUNTRY_ALIASES[q], None
    hits = sorted({c for n, c in name_map.items() if q in n})
    if len(hits) == 1:
        name = df[df["iso3"] == hits[0]]["country"].iloc[0]
        return hits[0], f"'{query}' matched uniquely by substring as {name} ({hits[0]}); use an ISO3 code if unintended"
    if len(hits) > 1:
        return None, f"'{query}' matches multiple countries {hits}; please use an ISO3 code"
    return None, f"Cannot identify country '{query}'"


def country_list(df):
    """83-country 'code=name' list (for error hints and --list-countries)"""
    names = df[["iso3", "country"]].drop_duplicates().sort_values("iso3")
    return "; ".join(f"{c}={n}" for c, n in zip(names["iso3"], names["country"]))
