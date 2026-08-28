# -*- coding: utf-8 -*-
"""
09_feature_select_delta.py - feature selection for the dTFR model

Input structure: [TFR0, 14 levels, 14 deltas, 29 masks] (n_feat groups = 14, TFR0 separate)
"Dropping" feature f = set its level, delta and both masks to missing (value 0, mask 1)
Greedy forward: start from the empty set (TFR0 only), add the feature with the highest
validation R2 until >= 99% x full model
Output: data/feature_selection_delta.json
Usage: python 09_feature_select_delta.py [--k 5]
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

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
MODELS = os.path.join(BASE, "models")


def predict_masked(X, layers, n_feat, keep):
    Xm = X.copy()
    # Structure: [TFR0, levels(14), deltas(14), masks(29)]
    L = 14
    drop = np.array([i for i in range(L) if i not in keep], dtype=int)
    off_v = 1 + L
    off_m = 1 + 2 * L
    Xm[:, off_v + drop] = 0.0
    Xm[:, off_v + L + drop] = 0.0
    Xm[:, off_m + drop] = 1.0
    Xm[:, off_m + L + drop] = 1.0
    acts = Xm
    for i, (W, b) in enumerate(layers):
        acts = acts @ W + b
        if i < len(layers) - 1:
            acts = np.maximum(acts, 0)
    return acts[:, 0]


def r2(y1, y2):
    ss = np.sum((y1 - np.mean(y1)) ** 2)
    return float(1.0 - np.sum((y1 - y2) ** 2) / ss if ss > 0 else np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()
    tag = f"delta_k{args.k}"

    z = np.load(os.path.join(DATA, f"splits_{tag}.npz"), allow_pickle=True)
    X_va, y_va, X_te, y_te = z["X_va"], z["y_va"], z["X_te"], z["y_te"]
    feats = z["feats"].tolist()
    base = feats[1:15]  # 14 features (drop tfr0)
    n_feat = 14

    w = np.load(os.path.join(MODELS, f"final_weights_{tag}.npz"), allow_pickle=True)
    layers = [([W, b]) for W, b in zip(w["Ws"], w["bs"])]

    full_r2 = r2(y_va, predict_masked(X_va, layers, n_feat, set(range(n_feat))))
    full_mae = float(np.mean(np.abs(predict_masked(X_va, layers, n_feat, set(range(n_feat))) - y_va)))
    print(f"  full model: valid R2={full_r2:.4f} MAE={full_mae:.4f}")

    # Permutation importance (shuffle level + delta + masks together)
    rng = np.random.default_rng(7)
    imp = {}
    for i, f in enumerate(base):
        Xp = X_va.copy()
        perm = rng.permutation(Xp.shape[0])
        for c in (1 + i, 1 + n_feat + i, 1 + 2 * n_feat + i, 1 + 3 * n_feat + i):
            Xp[:, c] = Xp[perm, c]
        imp[f] = float(full_r2 - r2(y_va, predict_masked(Xp, layers, n_feat, set(range(n_feat)))))
    order = sorted(imp.items(), key=lambda kv: -kv[1])
    print("\n  permutation importance (valid R2 drop):")
    for f, d in order:
        print(f"    {f:16s} dR2 = {d:+.4f}")

    # Greedy forward (starting from TFR0 only)
    keep, path = set(), []
    print(f"\n  greedy forward (target >= {full_r2*0.99:.4f}):")
    while len(keep) < n_feat:
        cand = [(r2(y_va, predict_masked(X_va, layers, n_feat, keep | {i})), f, i)
                for i, f in enumerate(base) if i not in keep]
        cand.sort(reverse=True)
        r, f, i = cand[0]
        keep.add(i)
        path.append([f, round(r, 4), round(r / full_r2, 4)])
        print(f"    +{f:16s} -> R2={r:.4f} ({r/full_r2:.1%})")
        if r >= full_r2 * 0.99:
            break

    yp = predict_masked(X_te, layers, n_feat, keep)
    te_r2 = r2(y_te, yp)
    te_mae = float(np.mean(np.abs(yp - y_te)))
    te_succ25 = float(np.mean(np.abs(yp - y_te) <= 0.25))
    full_te_r2 = r2(y_te, predict_masked(X_te, layers, n_feat, set(range(n_feat))))
    print(f"\n  minimal feature set ({len(keep)}): {[base[i] for i in sorted(keep)]}")
    print(f"  test: R2={te_r2:.4f} (full {full_te_r2:.4f}, {te_r2/full_te_r2:.1%})  MAE={te_mae:.4f}  |err|<=0.25={te_succ25:.2%}")

    out = {"full_val_r2": full_r2, "importance": dict(order),
           "greedy_path": path, "selected": [base[i] for i in sorted(keep)],
           "test_r2": te_r2, "test_full_r2": full_te_r2, "test_mae": te_mae,
           "test_succ25": te_succ25}
    with open(os.path.join(DATA, f"feature_selection_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"  -> data/feature_selection_{tag}.json")


if __name__ == "__main__":
    main()
