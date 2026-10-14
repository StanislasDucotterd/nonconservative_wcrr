#!/usr/bin/env python
"""Symmetry of the Jacobian of M(x) = V^T phi(W x) on FULL BSD68 images (321 x 481), without forming the Jacobian:
  ||J||_F^2 and ||(J - J^T)/2||_F^2 by Hutchinson's estimator with Gaussian probes v (J v by forward-mode AD, J^T v by reverse mode),
  ||J||_2 and ||J - J^T||_2 by power iteration on J^T J and on (J - J^T)^T (J - J^T).
Evaluated at the noisy input and at the denoising fixed point, for the natural images and for white noise of the same level.
    python scripts/jacobian_symmetry_full.py --device cuda:2 [--n-images 68 --probes 16 --power 40]"""
import argparse
import json
import math
import os
import sys
import torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ncr.trainer import load_checkpoint
from ncr.data import H5ImageDataset, fixed_noise


def make_ops(field, x):
    """J v by forward-mode AD (dual tensors) and J^T v by reverse-mode AD; plain autograd, no functorch closures (they held
    their graphs long enough to exhaust the GPU memory on full images)."""
    import torch.autograd.forward_ad as fwAD
    x = x.detach()

    def jvp(v):
        with fwAD.dual_level():
            out = field(fwAD.make_dual(x, v))
            return fwAD.unpack_dual(out).tangent.detach()

    def vjp(v):
        xg = x.clone().requires_grad_(True)
        return torch.autograd.grad(field(xg), xg, v)[0].detach()
    return jvp, vjp


def power_norm(apply, shape, iters, device, gen):
    """||A||_2 by power iteration on A^T A, apply(v) = A^T A v."""
    v = torch.randn(shape, device=device, dtype=torch.float64, generator=gen); v = v / v.norm()
    s = 0.0
    for _ in range(iters):
        w = apply(v); s = float(w.norm()); v = w / (s + 1e-30)
    return math.sqrt(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="trained_models/nonconservative_wcrr/last.pth")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-images", type=int, default=68)
    ap.add_argument("--probes", type=int, default=16)
    ap.add_argument("--power", type=int, default=40)
    ap.add_argument("--sigma", type=float, default=25.0)
    ap.add_argument("--out", default="results/jacobian_symmetry_full.json")
    ap.add_argument("--clean-only", action="store_true", help="only the clean natural images as evaluation point")
    args = ap.parse_args()
    torch.cuda.set_device(args.device)
    field, model, cfg = load_checkpoint(os.path.join(ROOT, args.checkpoint), args.device)
    field = field.double().eval(); model = model.double().eval()
    field.act.compile = False  # torch.compile + functorch transforms leak memory; plain autograd here
    ds = H5ImageDataset(os.path.join(ROOT, "data", "bsd68.h5"))
    gen = torch.Generator(device=args.device); gen.manual_seed(0)
    res = {}
    conditions = [(k, pt) for k in ("natural", "white noise") for pt in ("input", "fixed point")]
    if args.clean_only:
        conditions = [("natural", "clean")]
    for kind, point in conditions:
        if True:
            fro, spec = [], []
            for i in range(args.n_images):
                img = ds[i][None].to(args.device).double()
                noise = fixed_noise(img.shape, args.sigma / 255.0, 1234 + i, device=args.device).double()
                y = img + noise if kind == "natural" else img.mean() + noise
                if point == "input":
                    x = y
                elif point == "clean":
                    x = img
                else:
                    with torch.no_grad():
                        x = model(y, sigma=torch.full((1,), args.sigma, device=args.device, dtype=torch.float64))
                field.set_sigma(args.sigma)
                jvp, vjp = make_ops(field, x)
                num, den = 0.0, 0.0
                for _ in range(args.probes):
                    v = torch.randn(x.shape, device=args.device, dtype=torch.float64, generator=gen)
                    Jv, Jtv = jvp(v), vjp(v)
                    num += float((0.5 * (Jv - Jtv)).pow(2).sum()); den += float(Jv.pow(2).sum())
                fro.append(math.sqrt(num / den))
                nJ = power_norm(lambda v: vjp(jvp(v)), x.shape, args.power, args.device, gen)
                # (J - J^T)^T (J - J^T) v = (J^T - J)(J - J^T) v
                def ATA(v):
                    a = jvp(v) - vjp(v)
                    return vjp(a) - jvp(a)
                nA = power_norm(ATA, x.shape, args.power, args.device, gen)
                spec.append(0.5 * nA / nJ)
                torch.cuda.empty_cache()
            print(f"   memory {torch.cuda.memory_allocated() / 2**20:.0f} MB", flush=True)
            res[f"{kind} / {point}"] = {"asym_fro_mean": sum(fro) / len(fro), "asym_fro_min": min(fro), "asym_fro_max": max(fro),
                                        "asym_spec_mean": sum(spec) / len(spec), "asym_spec_min": min(spec), "asym_spec_max": max(spec)}
            r = res[f"{kind} / {point}"]
            print(f"{kind:12s} {point:12s}: asym Frobenius {r['asym_fro_mean']:.4f} [{r['asym_fro_min']:.4f}, {r['asym_fro_max']:.4f}]  spectral {r['asym_spec_mean']:.4f} [{r['asym_spec_min']:.4f}, {r['asym_spec_max']:.4f}]  ({args.n_images} full images, {args.probes} probes, {args.power} power iterations)", flush=True)
    json.dump(dict(checkpoint=args.checkpoint, n_images=args.n_images, probes=args.probes, power=args.power, sigma=args.sigma, results=res), open(os.path.join(ROOT, args.out), "w"), indent=1)


if __name__ == "__main__":
    main()
