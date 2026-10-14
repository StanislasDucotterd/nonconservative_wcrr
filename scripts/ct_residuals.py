#!/usr/bin/env python
"""Relative change of the SAGD iterates, ||x_{k+1} - x_k|| / ||x_k||, on one LoDoPaB test slice at the tuned (lambda, sigma) of
results/ct/<name>_<angles>/results.json, from the FBP, up to --max-iter iterations without stopping (Fig. 1 of the paper).
Run in the ODL / ASTRA environment, with the MFoE repository as for ct_reconstruction.py:
    python scripts/ct_residuals.py --mfoe-dir /path/to/MFoE --angles 60 --slice 0 --device cuda:0"""
import argparse
import json
import os
import sys
import torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from ct_reconstruction import mfoe_imports, load_split  # noqa: E402
from ncr.trainer import load_checkpoint  # noqa: E402
from ncr.solvers import solve_fixed_point  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="trained_models/nonconservative_wcrr/last.pth")
    ap.add_argument("--name", default="ours", help="results/ct/<name>_<angles>/results.json holds the tuned lambda and sigma")
    ap.add_argument("--mfoe-dir", default=os.path.expanduser("~/MFoE"))
    ap.add_argument("--angles", type=int, default=60)
    ap.add_argument("--slice", type=int, default=0)
    ap.add_argument("--max-iter", type=int, default=3000)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--stop", default="change", choices=["residual", "change"], help="quantity recorded per iteration")
    ap.add_argument("--out", default=None, help="default: results/ct/changes_<angles>.json")
    args = ap.parse_args()
    get_operators, _, ct_dir = mfoe_imports(args.mfoe_dir)
    dev = args.device
    torch.cuda.set_device(dev)
    fwd, fbp, bp = get_operators(img_size=362, space_range=1, num_angles=args.angles, det_shape=256, device=dev, fix_scaling=True)
    with torch.no_grad():
        u = torch.randn(1, 1, 362, 362, device=dev)
        for _ in range(60):
            u = bp(fwd(u)); data_lip = float(u.norm()); u = u / data_lip
    name, folder = load_split(ct_dir, "test_data")[args.slice]
    y = torch.load(os.path.join(folder, f"y_{args.angles}_256.pth"), weights_only=False).to(dev)
    x_gt = torch.load(os.path.join(folder, "image.pth"), weights_only=False).to(dev)
    r = json.load(open(os.path.join(ROOT, "results", "ct", f"{args.name}_{args.angles}", "results.json")))
    lamb, sigma = float(r["best_lamb"]), float(r["best_sigma"])
    field = load_checkpoint(os.path.join(ROOT, args.checkpoint), dev)[0].eval()
    field.set_sigma(sigma)
    L = float(field.lip_bound())
    alpha = 1.0 / (data_lip + lamb * L)
    G = lambda x, idx: bp(fwd(x) - y[idx]) + lamb * field(x)  # noqa: E731
    T = lambda x, idx: x - alpha * G(x, idx)  # noqa: E731
    with torch.no_grad():
        x, info = solve_fixed_point("agd", G, T, fbp(y), alpha, tol=1e-12, max_iter=args.max_iter, scale=bp(y).flatten(1).norm(dim=1), stop=args.stop)
        psnr = float(10 * torch.log10(1.0 / ((x.clamp(0, 1) - x_gt) ** 2).mean()))
    k = next((i + 1 for i, v in enumerate(info.history) if v < 1e-5), None)
    print(f"{args.name}: lambda {lamb:.4g}, sigma {sigma:.4g}, Lip {L:.1f}; below 1e-5 after {k} iterations; value after {len(info.history)} iterations {info.history[-1]:.2e}; PSNR {psnr:.3f}", flush=True)
    out = args.out or os.path.join(ROOT, "results", "ct", f"changes_{args.angles}.json")
    json.dump({"angles": args.angles, "slice": name, "max_iter": args.max_iter, "stop": args.stop,
               "models": {args.name: {"lamb": lamb, "sigma": sigma, "lip": L, "history": info.history, "psnr_final": psnr}}}, open(out, "w"))


if __name__ == "__main__":
    main()
