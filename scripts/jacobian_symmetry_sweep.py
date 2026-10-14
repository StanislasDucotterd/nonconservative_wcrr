#!/usr/bin/env python
"""Skew fraction s = ||(J - J^T)/2||_F / ||J||_F of the Jacobian of M along convex combinations of a natural image and Gaussian
white noise of the same scale:  x_t = (1 - t) img + t n,  n = mean(img) + white noise scaled to the l2 norm of img - mean(img).
Matrix-free (Hutchinson with Gaussian probes, forward-mode J v and reverse-mode J^T v), full BSD68 images, sigma = 25 in the model.
    python scripts/jacobian_symmetry_sweep.py --device cuda:1 --n-images 16 --probes 16"""
import argparse
import json
import math
import os
import sys
import torch
import torch.autograd.forward_ad as fwAD
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ncr.trainer import load_checkpoint
from ncr.data import H5ImageDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="trained_models/nonconservative_wcrr/last.pth")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-images", type=int, default=16)
    ap.add_argument("--probes", type=int, default=16)
    ap.add_argument("--n-t", type=int, default=11)
    ap.add_argument("--sigma", type=float, default=25.0)
    ap.add_argument("--out", default="results/jacobian_symmetry_sweep.json")
    args = ap.parse_args()
    torch.cuda.set_device(args.device)
    field, model, cfg = load_checkpoint(os.path.join(ROOT, args.checkpoint), args.device)
    field = field.double().eval(); field.act.compile = False
    field.set_sigma(args.sigma)
    ds = H5ImageDataset(os.path.join(ROOT, "data", "bsd68.h5"))
    gen = torch.Generator(device=args.device); gen.manual_seed(0)
    ts = [i / (args.n_t - 1) for i in range(args.n_t)]
    curves = []
    for i in range(args.n_images):
        img = ds[i][None].to(args.device).double()
        m = img.mean()
        eps = torch.randn(img.shape, device=args.device, dtype=torch.float64, generator=gen)
        n = m + eps * ((img - m).norm() / eps.norm())  # same mean, same l2 norm of the zero-mean part
        row = []
        for t in ts:
            x = (1 - t) * img + t * n
            num = den = 0.0
            for _ in range(args.probes):
                v = torch.randn(x.shape, device=args.device, dtype=torch.float64, generator=gen)
                with fwAD.dual_level():
                    Jv = fwAD.unpack_dual(field(fwAD.make_dual(x, v))).tangent.detach()
                xg = x.clone().requires_grad_(True)
                Jtv = torch.autograd.grad(field(xg), xg, v)[0].detach()
                num += float((0.5 * (Jv - Jtv)).pow(2).sum()); den += float(Jv.pow(2).sum())
            row.append(math.sqrt(num / den))
        curves.append(row)
        print(f"image {i:2d}: " + " ".join(f"{s:.3f}" for s in row), flush=True)
    C = torch.tensor(curves)
    print("t:     " + " ".join(f"{t:.1f}  " for t in ts))
    print("mean:  " + " ".join(f"{s:.3f}" for s in C.mean(0).tolist()))
    print("min:   " + " ".join(f"{s:.3f}" for s in C.min(0).values.tolist()))
    print("max:   " + " ".join(f"{s:.3f}" for s in C.max(0).values.tolist()))
    json.dump(dict(checkpoint=args.checkpoint, sigma=args.sigma, t=ts, curves=curves, mean=C.mean(0).tolist(), min=C.min(0).values.tolist(), max=C.max(0).values.tolist()),
              open(os.path.join(ROOT, args.out), "w"), indent=1)


if __name__ == "__main__":
    main()
