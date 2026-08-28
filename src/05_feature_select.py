# -*- coding: utf-8 -*-
"""
05_feature_select.py - feature selection: permutation importance + greedy minimal set

Idea (leveraging the missing-mask architecture):
  - A trained network's input = 13 feature values + 13 masks. "Dropping" a feature
    means setting its value to 0 (z-score mean) and its mask to 1, which is
    equivalent to the feature being fully unavailable -> the minimal-feature-set
    effect can be measured without retraining
  - Permutation importance: shuffle a feature's values and mask, measure the
    validation R2 drop
  - Greedy forward selection: start from the empty set, add the feature that
    maximizes validation R2 each step, until >= 99% of the full-model validation R2

Output: data/feature_selection.json (importance ranking + selected minimal set); prints everything
Usage: python 05_feature_select.py
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

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
MODELS = os.path.join(BASE, "models")


def predict_masked(X, layers, keep):
    """keep: set of feature indices to retain; dropped features get value 0 (mean) and mask 1"""
    n_feat = X.shape[1] // 2
    Xm = X.copy()
    drop = np.array([i for i in range(n_feat) if i not in keep], dtype=int)
    Xm[:, drop] = 0.0
    Xm[:, n_feat + drop] = 1.0
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
    z = np.load(os.path.join(DATA, "splits.npz"), allow_pickle=True)
    X_va, y_va, X_te, y_te = z["X_va"], z["y_va"], z["X_te"], z["y_te"]
    feats = z["feats"].tolist()
    n_feat = len(feats)

    w = np.load(os.path.join(MODELS, "final_weights.npz"), allow_pickle=True)
    layers = [([W, b]) for W, b in zip(w["Ws"], w["bs"])]

    full_r2 = r2(y_va, predict_masked(X_va, layers, set(range(n_feat))))
    full_mae = float(np.mean(np.abs(predict_masked(X_va, layers, set(range(n_feat))) - y_va)))
    print(f"  full model: valid R2={full_r2:.4f}  MAE={full_mae:.4f}")

    # Permutation importance (shuffle values and mask)
    rng = np.random.default_rng(7)
    imp = {}
    for i, f in enumerate(feats):
        Xp = X_va.copy()
        perm = rng.permutation(Xp.shape[0])
        Xp[:, i] = Xp[perm, i]
        Xp[:, n_feat + i] = Xp[perm, n_feat + i]
        r2p = r2(y_va, predict_masked(Xp, layers, set(range(n_feat))))
        imp[f] = float(full_r2 - r2p)
    order = sorted(imp.items(), key=lambda kv: -kv[1])
    print("\n  permutation importance (valid R2 drop):")
    for f, d in order:
        print(f"    {f:16s} dR2 = {d:+.4f}")

    # Greedy forward selection
    keep, best_r2, path = set(), -1e9, []
    print(f"\n  greedy forward selection (target: >= {full_r2*0.99:.4f} = 99% x full):")
    while len(keep) < n_feat:
        cand = []
        for i, f in enumerate(feats):
            if i in keep:
                continue
            k = keep | {i}
            r = r2(y_va, predict_masked(X_va, layers, k))
            cand.append((r, f, i))
        cand.sort(reverse=True)
        r, f, i = cand[0]
        keep.add(i)
        path.append([f, round(r, 4), round(r / full_r2, 4)])
        print(f"    +{f:16s} -> R2={r:.4f} ({r/full_r2:.1%} of full)")
        if r >= full_r2 * 0.99:
            break

    # Selected set performance on the test set
    te_r2 = r2(y_te, predict_masked(X_te, layers, keep))
    yp = predict_masked(X_te, layers, keep)
    te_mae = float(np.mean(np.abs(yp - y_te)))
    te_succ25 = float(np.mean(np.abs(yp - y_te) <= 0.25))
    full_te_r2 = r2(y_te, predict_masked(X_te, layers, set(range(n_feat))))
    print(f"\n  minimal feature set ({len(keep)}): {[feats[i] for i in sorted(keep)]}")
    print(f"  test: R2={te_r2:.4f} (full {full_te_r2:.4f}, ratio {te_r2/full_te_r2:.1%})  "
          f"MAE={te_mae:.4f}  |err|<=0.25={te_succ25:.2%}")

    out = {"full_val_r2": full_r2, "full_val_mae": full_mae,
           "importance": dict(order),
           "greedy_path": path,
           "selected": [feats[i] for i in sorted(keep)],
           "selected_idx": sorted(keep),
           "test_r2": te_r2, "test_full_r2": full_te_r2, "test_mae": te_mae,
           "test_succ25": te_succ25}
    with open(os.path.join(DATA, "feature_selection.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\n  -> data/feature_selection.json")


if __name__ == "__main__":
    main()
