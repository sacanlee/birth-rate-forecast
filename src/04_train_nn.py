# -*- coding: utf-8 -*-
"""
04_train_nn.py - pure-numpy MLP training (~11M params, CPU)

Architecture: [26 -> 2048 -> 2048 -> 2048 -> 1024 -> 512 -> 1]  ReLU + Dropout
Parameter count: ~11.2M

Training: Adam(lr=1e-3, weight_decay=1e-4) + Dropout(0.3) + early stopping
(validation MAE) + step lr schedule
Supports: --resume continues from checkpoint; checkpoint saved every 10 epochs
Evaluation: test R2 / MAE / success rate (|err| <= 0.25 and <= 0.30) / time extrapolation (>2010)

Usage: python 04_train_nn.py [--resume] [--epochs 800]
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

ARCH = [2048, 2048, 2048, 1024, 512]   # hidden layer widths
DROPOUT = 0.10   # experiments (60 epochs x 3 configs): 0.30 collapses deep nets / 0.15 = 0.726 R2 / 0.10 = 0.813 R2 best
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH = 512
EARLY_PATIENCE = 120
# Step decay (240-epoch experiments: step-lr converges better than cosine; adding a
# year feature hurts, so it is not used)
LR_SCHED = [(60, 1e-3), (120, 3e-4), (180, 1e-4), (260, 3e-5), (360, 1e-5)]
CKPT_EVERY = 10
SEED = 42


def init_net(d_in, arch, rng):
    """He initialization; returns [W, b] list"""
    layers = []
    sizes = [d_in] + arch + [1]
    for i in range(len(sizes) - 1):
        fan_in = sizes[i]
        W = rng.normal(0.0, np.sqrt(2.0 / fan_in), (fan_in, sizes[i + 1])).astype(np.float32)
        b = np.zeros(sizes[i + 1], dtype=np.float32)
        layers.append([W, b])
    return layers


def count_params(layers):
    return sum(int(W.size + b.size) for W, b in layers)


def forward(X, layers, drop=None, rng=None, train=True):
    """Return activations list and dropout masks; no dropout when train=False"""
    acts = [X]
    masks = []
    H = X
    for i, (W, b) in enumerate(layers):
        H = H @ W + b
        if i < len(layers) - 1:
            H = np.maximum(H, 0)
            if train and drop and drop > 0:
                m = (rng.random(H.shape) > drop).astype(np.float32) / (1 - drop)
                masks.append(m)
                H = H * m
            else:
                masks.append(None)
        acts.append(H)
    return acts, masks


def predict(X, layers):
    acts, _ = forward(X, layers, train=False)
    return acts[-1][:, 0]


def compute_grads(X, y, layers, drop, rng):
    acts, masks = forward(X, layers, drop, rng, train=True)
    grads = []
    # Output-layer error (MSE)
    dy = acts[-1][:, 0] - y
    for i in range(len(layers) - 1, -1, -1):
        W, b = layers[i]
        if i == len(layers) - 1:
            dH = dy[:, None]                     # output layer: gradient is wrt pre-activation
        else:
            dH = dH * (acts[i + 1] > 0)          # ReLU derivative (consistent with dropout zeros)
            if masks[i] is not None:
                dH = dH * masks[i]               # dropout mask (incl. 1/(1-p) scaling)
        dW = acts[i].T @ dH / X.shape[0]
        db = dH.mean(axis=0)
        grads.append((dW.astype(np.float32), db.astype(np.float32)))
        if i > 0:
            dH = dH @ W.T                        # propagate to previous layer
    grads.reverse()
    return grads


def mae(y1, y2):
    return float(np.mean(np.abs(y1 - y2)))


def r2(y1, y2):
    ss = np.sum((y1 - np.mean(y1)) ** 2)
    return float(1.0 - np.sum((y1 - y2) ** 2) / ss if ss > 0 else np.nan)


def success_rate(y1, y2, tol):
    return float(np.mean(np.abs(y1 - y2) <= tol))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--epochs", type=int, default=800)
    args = ap.parse_args()

    z = np.load(os.path.join(DATA, "splits.npz"), allow_pickle=True)
    X_tr, y_tr, X_va, y_va, X_te, y_te = z["X_tr"], z["y_tr"], z["X_va"], z["y_va"], z["X_te"], z["y_te"]
    Xt_tr, yt_tr, Xt_te, yt_te = z["Xt_tr"], z["yt_tr"], z["Xt_te"], z["yt_te"]
    feats = z["feats"].tolist()
    n_in = X_tr.shape[1]
    print(f"  input dim: {n_in} (13 features + 13 masks)  samples: {X_tr.shape[0]}/{X_va.shape[0]}/{X_te.shape[0]}")

    rng = np.random.default_rng(SEED)
    layers = init_net(n_in, ARCH, rng)
    n_param = count_params(layers)
    print(f"  parameters: {n_param:,} ({n_param/1e6:.1f}M)  architecture: {ARCH}")

    # Adam state
    m = [[np.zeros_like(W), np.zeros_like(b)] for W, b in layers]
    v = [[np.zeros_like(W), np.zeros_like(b)] for W, b in layers]
    epoch0, step, lr, best_mae, best_state = 0, 0, LR, None, None
    history = []

    ckpt_path = os.path.join(CKPT, "latest.npz")
    if args.resume and os.path.exists(ckpt_path):
        c = np.load(ckpt_path, allow_pickle=True)
        layers = [([W, b]) for W, b in zip(c["Ws"], c["bs"])]
        m = [([np.zeros_like(W), np.zeros_like(b)]) for W, b in layers]
        v = [([np.zeros_like(W), np.zeros_like(b)]) for W, b in layers]
        epoch0 = int(c["epoch"]); step = int(c["step"]); lr = float(c["lr"])
        best_mae = float(c["best_mae"])
        if c["history"].ndim:
            history = c["history"].tolist()
        print(f"  [resumed] epoch {epoch0}, best_mae {best_mae:.4f}")

    n_batch = int(np.ceil(len(X_tr) / BATCH))
    no_improve = 0
    t0 = time.time()
    print("  training started...")

    def _lr(epoch):
        lr = LR
        for cutoff, v in LR_SCHED:
            if epoch >= cutoff:
                lr = v
        return lr

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
                mh, vh = m[li][0], v[li][0]
                mhb, vhb = m[li][1], v[li][1]
                layers[li][0] -= lr * (mh / (np.sqrt(vh) + 1e-8) + WEIGHT_DECAY * layers[li][0])
                layers[li][1] -= lr * (mhb / (np.sqrt(vhb) + 1e-8))
            step += 1

        # Validate every epoch
        yp_va = predict(X_va, layers)
        vm, vr2 = mae(y_va, yp_va), r2(y_va, yp_va)
        history.append([epoch, vm, vr2, lr])
        if best_mae is None or vm < best_mae - 1e-5:
            best_mae = vm
            best_state = [([W.copy(), b.copy()]) for W, b in layers]
            no_improve = 0
        else:
            no_improve += 1

        if (epoch + 1) % 10 == 0 or no_improve == 0:
            print(f"  epoch {epoch+1}/{args.epochs}  val_MAE {vm:.4f}  val_R2 {vr2:.4f}"
                  f"  lr {lr:.2e}  ({time.time()-t0:.0f}s)")

        if (epoch + 1) % CKPT_EVERY == 0:
            np.savez(ckpt_path,
                     Ws=np.array([W for W, _ in layers], dtype=object),
                     bs=np.array([b for _, b in layers], dtype=object),
                     epoch=epoch + 1, step=step, lr=lr, best_mae=best_mae or 0,
                     history=np.array(history), allow_pickle=True)
        if no_improve >= EARLY_PATIENCE:
            print(f"  early stop at epoch {epoch+1}")
            break

    # Restore best
    if best_state is not None:
        layers = best_state
    np.savez(ckpt_path,
             Ws=np.array([W for W, _ in layers], dtype=object),
             bs=np.array([b for _, b in layers], dtype=object),
             epoch=args.epochs, step=step, lr=lr, best_mae=best_mae or 0,
             history=np.array(history), allow_pickle=True)
    with open(os.path.join(MODELS, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"arch": ARCH, "n_params": n_param, "dropout": DROPOUT,
                   "lr0": LR, "weight_decay": WEIGHT_DECAY, "batch": BATCH,
                   "seed": SEED, "epochs": epoch + 1, "best_val_mae": best_mae,
                   "features": feats}, f, ensure_ascii=False, indent=1)

    # Final evaluation
    print("\n== Test evaluation (random split) ==")
    for name, X, y in [("train", X_tr, y_tr), ("valid", X_va, y_va), ("test", X_te, y_te)]:
        yp = predict(X, layers)
        print(f"  {name}: MAE {mae(y, yp):.4f}  R2 {r2(y, yp):.4f}  "
              f"|err|<=0.25 {success_rate(y, yp, 0.25):.4f}  |err|<=0.30 {success_rate(y, yp, 0.30):.4f}")
    print("== Time extrapolation (train <=2010) ==")
    yp = predict(Xt_te, layers)
    print(f"  2011-2024: MAE {mae(yt_te, yp):.4f}  R2 {r2(yt_te, yp):.4f}  "
          f"|err|<=0.25 {success_rate(yt_te, yp, 0.25):.4f}")

    np.savez(os.path.join(MODELS, "final_weights.npz"),
             Ws=np.array([W for W, _ in layers], dtype=object),
             bs=np.array([b for _, b in layers], dtype=object),
             feats=np.array(feats, dtype=object),
             test_mae=mae(y_te, predict(X_te, layers)),
             test_r2=r2(y_te, predict(X_te, layers)), allow_pickle=True)
    with open(os.path.join(MODELS, "train_history.csv"), "w", encoding="utf-8") as f:
        f.write("epoch,val_mae,val_r2,lr\n")
        for h in history:
            f.write(f"{h[0]},{h[1]:.6f},{h[2]:.6f},{h[3]:.2e}\n")
    print("  model saved: models/final_weights.npz, config.json, train_history.csv")
    print(f"  total time {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
