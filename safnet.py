"""
SAFNet — Siamese Adaptive Fusion Network (PyTorch) for SAR change detection.

This module contains the SAFNet model, training loop, and inference functions,
extracted from the main analysis script for modularity.
"""
import os
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False


def normalize_db(db):
    """dB -> [0,1] for the network; NaN/nodata -> 0 (mapped to the -30 dB floor).
    Returns (x float32, valid_mask bool)."""
    valid = np.isfinite(db)
    x = np.clip(np.where(valid, db, -30.0), -30.0, 0.0)
    x = (x + 30.0) / 30.0
    return x.astype("float32"), valid


def sample_patches(stack_db, rng, patch_size, patches_per_scene):
    """Random, mostly-valid PATCHxPATCH crops from every scene (normalized)."""
    pool = []
    for db in stack_db:
        x, valid = normalize_db(db)
        H, W = x.shape
        if H < patch_size or W < patch_size:
            continue
        for _ in range(patches_per_scene):
            i = int(rng.integers(0, H - patch_size + 1))
            j = int(rng.integers(0, W - patch_size + 1))
            if valid[i:i+patch_size, j:j+patch_size].mean() < 0.5:
                continue
            pool.append(x[i:i+patch_size, j:j+patch_size].copy())
    if not pool:
        raise RuntimeError("no valid training patches (scenes smaller than PATCH?)")
    return pool


def synth_change_batch(pool, rng, n, patch_size):
    """Build n self-supervised (A, B, mask) triples with NO real labels."""
    A, B, M = [], [], []
    P = patch_size
    for _ in range(n):
        p = pool[int(rng.integers(len(pool)))]
        a = p + rng.normal(0, 0.05, p.shape).astype("float32")
        m = np.zeros((P, P), "float32")
        if rng.random() < 0.5:
            b = p + rng.normal(0, 0.05, p.shape).astype("float32")
        else:
            b = p.copy()
            q = pool[int(rng.integers(len(pool)))]
            rh = int(rng.integers(P // 6, P // 2)); rw = int(rng.integers(P // 6, P // 2))
            i = int(rng.integers(0, P - rh + 1)); j = int(rng.integers(0, P - rw + 1))
            b[i:i+rh, j:j+rw] = q[i:i+rh, j:j+rw]
            b = b + rng.normal(0, 0.05, p.shape).astype("float32")
            m[i:i+rh, j:j+rw] = 1.0
        A.append(np.clip(a, 0, 1)); B.append(np.clip(b, 0, 1)); M.append(m)
    t = lambda arr: torch.from_numpy(np.stack(arr))[:, None]
    return t(A), t(B), t(M)


class SAFNet(nn.Module):
    """Siamese Adaptive Fusion Network."""
    def __init__(self, base=16, fuse=16):
        super().__init__()
        def block(ci, co):
            return nn.Sequential(nn.Conv2d(ci, co, 3, padding=1), nn.ReLU(True),
                                 nn.Conv2d(co, co, 3, padding=1), nn.ReLU(True))
        self.e1 = block(1, base)
        self.e2 = block(base, base * 2)
        self.e3 = block(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.dec = nn.Sequential(nn.Conv2d(base * 4, base * 2, 3, padding=1), nn.ReLU(True),
                                 nn.Conv2d(base * 2, base, 3, padding=1), nn.ReLU(True),
                                 nn.Conv2d(base, 1, 3, padding=1))
        self.p1 = nn.Conv2d(base, fuse, 1)
        self.p2 = nn.Conv2d(base * 2, fuse, 1)
        self.p3 = nn.Conv2d(base * 4, fuse, 1)
        self.att = nn.Sequential(nn.Conv2d(fuse * 3, fuse, 1), nn.ReLU(True),
                                 nn.Conv2d(fuse, 3, 1))
        self.head = nn.Sequential(nn.Conv2d(fuse, fuse, 3, padding=1), nn.ReLU(True),
                                  nn.Conv2d(fuse, 1, 1))

    def encode(self, x):
        f1 = self.e1(x)
        f2 = self.e2(self.pool(f1))
        f3 = self.e3(self.pool(f2))
        return f1, f2, f3

    def reconstruct(self, x):
        _, _, f3 = self.encode(x)
        up = F.interpolate(f3, size=x.shape[-2:], mode="nearest")
        return self.dec(up)

    def change_logits(self, a, b):
        fa, fb = self.encode(a), self.encode(b)
        hw = a.shape[-2:]
        d1 = self.p1((fa[0] - fb[0]).abs())
        d2 = F.interpolate(self.p2((fa[1] - fb[1]).abs()), size=hw, mode="bilinear", align_corners=False)
        d3 = F.interpolate(self.p3((fa[2] - fb[2]).abs()), size=hw, mode="bilinear", align_corners=False)
        w = torch.softmax(self.att(torch.cat([d1, d2, d3], 1)), dim=1)
        fused = w[:, 0:1] * d1 + w[:, 1:2] * d2 + w[:, 2:3] * d3
        return self.head(fused)


def train_safnet(stack_db, rng, C):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(C['SEED'])
    pool = sample_patches(stack_db, rng, C['PATCH'], C['PATCHES_PER_SCENE'])
    Xrec = torch.from_numpy(np.stack(pool))[:, None]
    n = Xrec.shape[0]
    print(f"[safnet] training on {n} patches ({C['PATCH']}x{C['PATCH']}) for {C['EPOCHS']} epochs on {device}...")

    model = SAFNet(base=C['BASE'], fuse=C['FUSE']).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=C['LR'])
    mse, bce = nn.MSELoss(), nn.BCEWithLogitsLoss()
    model.train()

    for ep in range(C['EPOCHS']):
        perm = rng.permutation(n)
        r_tot = c_tot = 0.0
        for k in range(0, n, C['BATCH']):
            idx = perm[k:k+C['BATCH']]
            b_rec = Xrec[torch.from_numpy(idx)]
            A, B, M = synth_change_batch(pool, rng, len(idx), C['PATCH'])
            opt.zero_grad(set_to_none=True)
            l_rec = mse(model.reconstruct(b_rec.to(device)), b_rec.to(device))
            l_chg = bce(model.change_logits(A.to(device), B.to(device)), M.to(device))
            (l_rec + l_chg).backward(); opt.step()
            r_tot += l_rec.item() * len(idx); c_tot += l_chg.item() * len(idx)
        if ep % 3 == 0 or ep == C['EPOCHS'] - 1:
            print(f"          epoch {ep:2d}/{C['EPOCHS']-1}  recon MSE = {r_tot/n:.4f}  change BCE = {c_tot/n:.4f}")
    model.eval()
    return model


def safnet_change_prob(model, baseline_db, latest_db):
    """Per-pixel change probability from the adaptive-fusion head."""
    xb, vb = normalize_db(baseline_db)
    xl, vl = normalize_db(latest_db)
    device = next(model.parameters()).device
    with torch.no_grad():
        A = torch.from_numpy(xb)[None, None].to(device)
        B = torch.from_numpy(xl)[None, None].to(device)
        prob = torch.sigmoid(model.change_logits(A, B))[0, 0].cpu().numpy()
    valid = vb & vl
    return np.where(valid, prob, np.nan), valid