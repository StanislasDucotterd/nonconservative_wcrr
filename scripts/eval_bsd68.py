#!/usr/bin/env python
"""Per-image BSD68 denoising PSNRs of trained runs, with the trainer's validation protocol (fixed noise, seed
1234 + i + 100000 k for the k-th sigma of [25, 5, 15], output clamped to [0, 1]), saved to results/bsd68/<run>.json, and
paired comparisons between runs (mean difference, its standard error, paired t and the number of images won).
    python scripts/eval_bsd68.py --device cuda:0            # the paper's model, trained_models/nonconservative_wcrr/last.pth
    python scripts/eval_bsd68.py --compare [--runs a b]        # paired tables of the saved results, no evaluation"""
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
from ncr.utils import psnr

OUT = os.path.join(ROOT, "results", "bsd68")
SIGMAS = [25.0, 5.0, 15.0]  # the trainer's val_sigmas order fixes the noise seeds


def evaluate(run, device, sigmas, ckpt="last.pth", seed=1234, solver=None, tol=None, max_iter=None, stop=None, suffix=""):
    field, model, cfg = load_checkpoint(os.path.join(ROOT, "trained_models", run, ckpt), device)
    if solver: model.solver = solver  # override the checkpoint's training solver for the evaluation
    if tol: model.tol = tol
    if max_iter: model.max_iter = max_iter
    if stop: model.stop = stop
    model.eval()
    ds = H5ImageDataset(os.path.join(ROOT, "data", "bsd68.h5"))
    out = {"run": run, "checkpoint": ckpt, "solver": {"solver": model.solver, "tol": model.tol, "max_iter": model.max_iter, "stop": model.stop}, "psnr": {}, "iters": {}}
    with torch.no_grad():
        for s in sigmas:
            k = SIGMAS.index(s)
            ps, its = [], []
            for i in range(len(ds)):
                x = ds[i][None].to(device)
                y = x + fixed_noise(x.shape, s / 255.0, seed + i + 100000 * k, device=device)
                z = model(y, sigma=torch.full((1,), s, device=device)).clamp(0, 1)
                ps.append(float(psnr(z, x).mean())); its.append(float(model.stats().get("fp_iters_mean", 0.0)))
            out["psnr"][f"{s:g}"] = ps; out["iters"][f"{s:g}"] = its
            print(f"{run} sigma {s:g}: {sum(ps)/len(ps):.3f} dB ({sum(its)/len(its):.0f} iterations/image)", flush=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, f"{run}{suffix}.json"), "w"), indent=1)
    return out


def compare(runs=None):
    files = sorted(f for f in os.listdir(OUT) if f.endswith(".json"))
    res = {json.load(open(os.path.join(OUT, f)))["run"]: json.load(open(os.path.join(OUT, f))) for f in files}
    runs = runs or list(res)
    for s in [k for k in ("25", "15", "5") if all(k in res[r]["psnr"] for r in runs)]:
        print(f"\nsigma {s}: mean PSNR " + ", ".join(f"{r} {sum(res[r]['psnr'][s])/len(res[r]['psnr'][s]):.3f}" for r in runs))
        print(f"  paired A - B over {len(res[runs[0]]['psnr'][s])} images: mean (std. error) [paired t] A wins / ties / B wins")
        for a in range(len(runs)):
            for b in range(a + 1, len(runs)):
                da = torch.tensor(res[runs[a]]["psnr"][s]) - torch.tensor(res[runs[b]]["psnr"][s])
                n = da.numel(); se = da.std() / math.sqrt(n); t = float(da.mean() / se) if se > 0 else float("inf")
                print(f"  {runs[a]:36s} - {runs[b]:36s}: {da.mean():+.3f} ({se:.3f}) [t {t:+.1f}]  {int((da > 0.005).sum())} / {int((da.abs() <= 0.005).sum())} / {int((da < -0.005).sum())}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=["nonconservative_wcrr"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--sigmas", nargs="*", type=float, default=SIGMAS)
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--ckpt", default="last.pth", help="checkpoint file inside the run folder (last.pth = final epoch)")
    ap.add_argument("--solver", default=None, help="override the training solver, e.g. agd (SAGD)")
    ap.add_argument("--stop", default=None, choices=[None, "residual", "change"])
    ap.add_argument("--tol", type=float, default=None)
    ap.add_argument("--max-iter", type=int, default=None)
    ap.add_argument("--suffix", default="", help="appended to the output file name")
    args = ap.parse_args()
    if args.runs and not args.compare:  # --compare alone only reads the saved results (optionally restricted to --runs)
        torch.cuda.set_device(args.device)
        for r in args.runs:
            evaluate(r, args.device, args.sigmas, ckpt=args.ckpt, solver=args.solver, tol=args.tol, max_iter=args.max_iter, stop=args.stop, suffix=args.suffix)
    compare(args.runs or None)


if __name__ == "__main__":
    main()
