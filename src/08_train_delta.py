# -*- coding: utf-8 -*-
"""
08_train_delta.py - train the dTFR neural network (user follow-up requirement 2)

Target: dTFR(t -> t+k) = TFR(t+k) - TFR(t)
Input:  [TFR(t), X(t) levels, dX, masks]  (58 dims)
Architecture: same as 04: [58 -> 2048 -> 2048 -> 2048 -> 1024 -> 512 -> 1] ~11.1M params
Training: Adam + dropout 0.10 + step lr + early stopping
Evaluation: dTFR metrics + TFR-level metrics (chained rolling) + baselines (d=0 persistence,
            linear regression)

Usage: python 08_train_delta.py [--k 5] [--epochs 400] [--resume]
"""
import argparse
import json
import os
import sys
import time

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
MODELS = os.path.join(BASE, "models")
CKPT = os.path.join(MODELS, "checkpoints")
os.makedirs(CKPT, exist_ok=True)

ARCH = [2048, 2048, 2048, 1024, 512]
ARCH_BIG = [3072, 3072, 3072, 2048, 1024, 512]   # ~28M params (optional ~30M)
DROPOUT = 0.10
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH = 512
EARLY_PATIENCE = 120
CKPT_EVERY = 10
SEED = 42
LR_SCHED = [(60, 1e-3), (120, 3e-4), (180, 1e-4), (260, 3e-5), (360, 1e-5)]


def init_net(d_in, arch, rng):
    layers = []
    sizes = [d_in] + arch + [1]
    for i in range(len(sizes) - 1):
        W = rng.normal(0.0, np.sqrt(2.0 / sizes[i]), (sizes[i], sizes[i + 1])).astype(np.float32)
        layers.append([W, np.zeros(sizes[i + 1], dtype=np.float32)])
    return layers


def count_params(layers):
    return sum(int(W.size + b.size) for W, b in layers)


def forward(X, layers, drop, rng, train=True):
    acts, masks = [X], []
    H = X
    for i, (W, b) in enumerate(layers):
        H = H @ W + b
        if i < len(layers) - 1:
            H = np.maximum(H, 0)
            if train and drop > 0:
                m = (rng.random(H.shape) > drop).astype(np.float32) / (1 - drop)
                masks.append(m)
                H = H * m
            else:
                masks.append(None)
        acts.append(H)
    return acts, masks


def predict(X, layers):
    acts, _ = forward(X, layers, 0, None, train=False)
    return acts[-1][:, 0]


def compute_grads(X, y, layers, drop, rng):
    acts, masks = forward(X, layers, drop, rng, train=True)
    grads = []
    dy = acts[-1][:, 0] - y
    for i in range(len(layers) - 1, -1, -1):
        W, b = layers[i]
        if i == len(layers) - 1:
            dH = dy[:, None]
        else:
            dH = dH * (acts[i + 1] > 0)
            if masks[i] is not None:
                dH = dH * masks[i]
        dW = acts[i].T @ dH / X.shape[0]
        db = dH.mean(axis=0)
        grads.append((dW.astype(np.float32), db.astype(np.float32)))
        if i > 0:
            dH = dH @ W.T
    grads.reverse()
    return grads


def mae(y1, y2):
    return float(np.mean(np.abs(y1 - y2)))


def r2(y1, y2):
    ss = np.sum((y1 - np.mean(y1)) ** 2)
    return float(1.0 - np.sum((y1 - y2) ** 2) / ss if ss > 0 else np.nan)


def succ(y1, y2, tol):
    return float(np.mean(np.abs(y1 - y2) <= tol))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--big", action="store_true", help="30M-param architecture (~28M)")
    args = ap.parse_args()
    k = args.k
    arch = ARCH_BIG if args.big else ARCH
    tag = f"delta_k{k}{'_big' if args.big else ''}"

    z = np.load(os.path.join(DATA, f"splits_{tag}.npz"), allow_pickle=True)
    X_tr, y_tr, X_va, y_va, X_te, y_te = (z["X_tr"], z["y_tr"], z["X_va"], z["y_va"], z["X_te"], z["y_te"])
    Xt_tr, yt_tr, Xt_te, yt_te = z["Xt_tr"], z["yt_tr"], z["Xt_te"], z["yt_te"]
    meta = z["meta"]
    n_in = X_tr.shape[1]
    print(f"  k={k} input dim {n_in} samples {X_tr.shape[0]}/{X_va.shape[0]}/{X_te.shape[0]} "
          f"time {Xt_tr.shape[0]}/{Xt_te.shape[0]}")

    # ===== Baseline 1: d=0 persistence; Baseline 2: linear regression =====
    def lin_eval(Xa, ya, Xb, yb, name):
        A = np.concatenate([np.ones((Xa.shape[0], 1)), Xa], axis=1)
        beta, *_ = np.linalg.lstsq(A, ya, rcond=None)
        B = np.concatenate([np.ones((Xb.shape[0], 1)), Xb], axis=1)
        yp = B @ beta
        print(f"  [linear baseline {name}] MAE {mae(yb, yp):.4f} R2 {r2(yb, yp):.4f} |err|<=0.25 {succ(yb, yp, 0.25):.2%}")
        return beta

    print(f"  [d=0 persistence] time test MAE {mae(yt_te, np.zeros_like(yt_te)):.4f}")
    lin_eval(Xt_tr, yt_tr, Xt_te, yt_te, "time test")

    rng = np.random.default_rng(SEED)
    layers = init_net(n_in, arch, rng)
    print(f"  parameters: {count_params(layers):,} ({count_params(layers)/1e6:.1f}M)")

    m = [[np.zeros_like(W), np.zeros_like(b)] for W, b in layers]
    v = [[np.zeros_like(W), np.zeros_like(b)] for W, b in layers]
    epoch0, step, best_mae, best_state, history = 0, 0, None, None, []
    ckpt_path = os.path.join(CKPT, f"latest_{tag}.npz")
    if args.resume and os.path.exists(ckpt_path):
        c = np.load(ckpt_path, allow_pickle=True)
        layers = [([W, b]) for W, b in zip(c["Ws"], c["bs"])]
        epoch0 = int(c["epoch"]); step = int(c["step"])
        best_mae = float(c["best_mae"])
        if c["history"].ndim:
            history = c["history"].tolist()
        print(f"  [resumed] epoch {epoch0}, best_mae {best_mae:.4f}")

    def _lr(epoch):
        lr = LR
        for cutoff, val in LR_SCHED:
            if epoch >= cutoff:
                lr = val
        return lr

    n_batch = int(np.ceil(len(X_tr) / BATCH))
    no_improve, t0 = 0, time.time()
    print("  training started...")
    for epoch in range(epoch0, args.epochs):
        lr = _lr(epoch)
        perm = rng.permutation(len(X_tr))
        for bi in range(n_batch):
            idx = perm[bi * BATCH:(bi + 1) * BATCH]
            g = compute_grads(X_tr[idx], y_tr[idx], layers, DROPOUT, rng)
            for li, (gW, gb) in enumerate(g):
                m[li][0] = 0.9 * m[li][0] + 0.1 * gW
                m[li][1] = 0.9 * m[li][1] + 0.1 * gb
                v[li][0] = 0.999 * v[li][0] + 0.001 * gW * gW
                v[li][1] = 0.999 * v[li][1] + 0.001 * gb * gb
                layers[li][0] -= lr * (m[li][0] / (np.sqrt(v[li][0]) + 1e-8) + WEIGHT_DECAY * layers[li][0])
                layers[li][1] -= lr * (m[li][1] / (np.sqrt(v[li][1]) + 1e-8))
            step += 1
        yp = predict(X_va, layers)
        vm = mae(y_va, yp)
        history.append([epoch, vm, r2(y_va, yp), lr])
        if best_mae is None or vm < best_mae - 1e-5:
            best_mae = vm
            best_state = [([W.copy(), b.copy()]) for W, b in layers]
            no_improve = 0
        else:
            no_improve += 1
        if (epoch + 1) % 10 == 0 or no_improve == 0:
            print(f"  ep {epoch+1} val_MAE {vm:.4f} val_R2 {r2(y_va, yp):.4f} ({time.time()-t0:.0f}s)")
        if (epoch + 1) % CKPT_EVERY == 0:
            np.savez(ckpt_path, Ws=np.array([W for W, _ in layers], dtype=object),
                     bs=np.array([b for _, b in layers], dtype=object),
                     epoch=epoch + 1, step=step, best_mae=best_mae or 0,
                     history=np.array(history), allow_pickle=True)
        if no_improve >= EARLY_PATIENCE:
            print(f"  early stop at epoch {epoch+1}")
            break

    if best_state is not None:
        layers = best_state
    np.savez(ckpt_path, Ws=np.array([W for W, _ in layers], dtype=object),
             bs=np.array([b for _, b in layers], dtype=object),
             epoch=args.epochs, step=step, best_mae=best_mae or 0,
             history=np.array(history), allow_pickle=True)

    # ===== Evaluation =====
    def ev(name, X, y, tfr0):
        yp = predict(X, layers)
        print(f"  [{name}] dTFR: MAE {mae(y, yp):.4f} R2 {r2(y, yp):.4f} "
              f"|err|<=0.25 {succ(y, yp, 0.25):.2%} |err|<=0.50 {succ(y, yp, 0.50):.2%}")
        if tfr0 is not None:
            tfr_p = tfr0 + yp
            tfr_t = tfr0 + y
            print(f"  [{name}] TFR level: MAE {mae(tfr_t, tfr_p):.4f} R2 {r2(tfr_t, tfr_p):.4f} "
                  f"|err|<=0.25 {succ(tfr_t, tfr_p, 0.25):.2%}")
        return yp

    print("\n== dTFR evaluation ==")
    tfr0_tr = np.array([float(r[3]) for r in meta[:len(X_tr)]])
    ev("train", X_tr, y_tr, tfr0_tr)
    ev("valid", X_va, y_va, None)
    ev("test (random)", X_te, y_te, None)
    # TFR level for the time test (time test = window start year >= 2011, meta column 1)
    tfr0_te = np.array([float(r[3]) for r in meta])
    mask_te = (np.array([int(r[1]) for r in meta]) >= 2011)
    ev("test (time, >=2011)", Xt_te, yt_te, tfr0_te[mask_te])

    # Chained rolling: roll every k years from 2010 (simplified to single steps
    # using the actual structure of the time-test sample)
    print("\n== Chained evaluation (2010->2025, k=5, rolling windows) ==")
    meta_arr = np.array(meta, dtype=object)
    # Xt_te holds the rows of the full set filtered by year >= 2011; i is the full-set
    # index, so map it to the Xt_te row number j
    te_mask = np.array([int(r[1]) for r in meta_arr]) >= 2011
    j_of = {i: int(np.sum(te_mask[:i])) for i in range(len(meta_arr)) if te_mask[i]}
    for iso in ["CHN", "JPN", "IND", "NGA", "USA", "KOR", "BRA", "MEX"]:
        sel = [(i, r) for i, r in enumerate(meta_arr)
               if r[0] == iso and 2010 <= int(r[1]) <= 2025 and te_mask[i]]
        if not sel:
            continue
        line = f"  {iso}: "
        for i, r in sorted(sel, key=lambda x: int(x[1][1])):
            yp = predict(Xt_te[[j_of[i]]], layers)[0]
            line += f"{int(r[1])}-{int(r[2])}:d{float(r[4]):+.2f}(true)/{yp:+.2f}(pred)  "
        print(line)

    np.savez(os.path.join(MODELS, f"final_weights_{tag}.npz"),
             Ws=np.array([W for W, _ in layers], dtype=object),
             bs=np.array([b for _, b in layers], dtype=object),
             allow_pickle=True)
    with open(os.path.join(MODELS, f"config_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump({"arch": arch, "n_params": count_params(layers), "dropout": DROPOUT,
                   "lr0": LR, "weight_decay": WEIGHT_DECAY, "batch": BATCH,
                   "epochs": epoch + 1, "best_val_mae": best_mae, "k": k}, f, ensure_ascii=False, indent=1)
    with open(os.path.join(MODELS, f"history_{tag}.csv"), "w", encoding="utf-8") as f:
        f.write("epoch,val_mae,val_r2,lr\n")
        for h in history:
            f.write(f"{h[0]},{h[1]:.6f},{h[2]:.6f},{h[3]:.2e}\n")
    print(f"  model saved: models/final_weights_{tag}.npz + config_{tag}.json")
    print(f"  total time {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
