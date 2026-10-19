#!/usr/bin/env python
"""The numbers quoted in the paper for one checkpoint, in one JSON file:
  * filter pairs: aligned / flipped / in-between counts (cos > 0.9, cos < -0.5), cosine range of the flipped pairs, the fraction
    of ||v_c|| orthogonal to w_c (mean over aligned and over flipped pairs), the range of ||v_c|| / ||w_c||;
  * the spline: slope on the first cell, peak value and position, phi(0.09) / peak, where psi stops being convex, min slope,
    mu and the Lipschitz bound mu * max |phi'|;
  * the skew fraction s = ||(J - J^T)/2||_F / ||J||_F at x = 0 on a full-size image (linear regime, J = mu phi'(0) V^T W),
    matrix-free with Gaussian probes (forward-mode jvp, reverse-mode vjp);
  * the extreme eigenvalues of Sym J_M at the denoising fixed points (sigma = 25) of full BSD68 images, matrix-free by shifted
    power iterations; the Rayleigh quotient reached by the shifted iteration is an upper bound on the smallest eigenvalue, so a
    negative value certifies that M is not monotone there.
    python scripts/paper_numbers.py --checkpoint trained_models/<run>/last.pth --device cuda:0 [--out results/paper_numbers.json]"""
import argparse
import json
import math
import os
import sys
import numpy as np
import torch
import torch.autograd.forward_ad as fwAD
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from ncr.trainer import load_checkpoint  # noqa: E402
from ncr.data import H5ImageDataset, fixed_noise  # noqa: E402
from jacobian_symmetry_full import make_ops  # noqa: E402  (matrix-free J v and J^T v on full images)


def skew_fraction_at(field, x, probes, gen):
    num, den = 0.0, 0.0
    for _ in range(probes):
        v = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=gen)
        with fwAD.dual_level():
            Jv = fwAD.unpack_dual(field(fwAD.make_dual(x, v))).tangent.detach()
        xg = x.clone().requires_grad_(True)
        Jtv = torch.autograd.grad(field(xg), xg, v)[0].detach()
        num += float((0.5 * (Jv - Jtv)).pow(2).sum()); den += float(Jv.pow(2).sum())
    return math.sqrt(num / den)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="trained_models/nonconservative_wcrr/last.pth")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--sigma", type=float, default=25.0)
    ap.add_argument("--probes", type=int, default=16)
    ap.add_argument("--n-images", type=int, default=16, help="images for the eigenvalue analysis at denoising fixed points")
    ap.add_argument("--power", type=int, default=200, help="power iterations per eigenvalue")
    ap.add_argument("--out", default="results/paper_numbers.json")
    args = ap.parse_args()
    dev = args.device
    if dev.startswith("cuda"):
        torch.cuda.set_device(dev)
    field, model, cfg = load_checkpoint(os.path.join(ROOT, args.checkpoint), dev)
    field = field.double().eval(); model = model.double().eval()
    field.act.compile = False
    out = {"checkpoint": args.checkpoint, "epoch": int(torch.load(os.path.join(ROOT, args.checkpoint), map_location="cpu", weights_only=False).get("epoch", -1)),
           "denoiser": cfg.get("denoiser", {})}

    # ---- filter pairs
    with torch.no_grad():
        W = field.W.impulse().double(); V = field.V_impulse().double()
    w, v = W.flatten(1), V.flatten(1)
    nw, nv = w.norm(dim=1), v.norm(dim=1)
    cos = (w * v).sum(1) / (nw * nv)
    orth = torch.sqrt((1 - cos ** 2).clamp(min=0))  # ||v_c - <v_c, w_c/||w_c||> w_c/||w_c|| || / ||v_c||
    aligned, flipped = cos > 0.9, cos < -0.5
    other = ~(aligned | flipped)
    out["pairs"] = {"n": int(cos.numel()), "aligned": int(aligned.sum()), "flipped": int(flipped.sum()), "other": int(other.sum()),
                    "cos_flipped_range": [float(cos[flipped].min()), float(cos[flipped].max())] if flipped.any() else None,
                    "cos_aligned_range": [float(cos[aligned].min()), float(cos[aligned].max())] if aligned.any() else None,
                    "cos_other": cos[other].tolist(),
                    "orth_fraction_aligned_mean": float(orth[aligned].mean()) if aligned.any() else None,
                    "orth_fraction_flipped_mean": float(orth[flipped].mean()) if flipped.any() else None,
                    "orth_fraction_aligned_range": [float(orth[aligned].min()), float(orth[aligned].max())] if aligned.any() else None,
                    "orth_fraction_flipped_range": [float(orth[flipped].min()), float(orth[flipped].max())] if flipped.any() else None,
                    "norm_ratio_range": [float((nv / nw).min()), float((nv / nw).max())]}

    # ---- spline
    act = field.act
    with torch.no_grad():
        c = act.coefficients()[0].cpu().numpy(); t = (np.arange(act.K) - act.mid) * float(act.h)
        mu = float(torch.exp(act.log_mu)) if hasattr(act, "log_mu") else None
    pos = t >= 0; tp, cp = t[pos], c[pos]
    slopes = np.diff(cp) / float(act.h)
    k = int(cp.argmax())
    # psi convex while phi increases: first knot after which the slope is negative
    first_neg = int(np.argmax(slopes < 0)) if (slopes < 0).any() else None
    out["spline"] = {"knots": int(act.K), "span": float(act.span), "h": float(act.h), "mu": mu,
                     "slope_first_cell": float(slopes[0]), "slope_min": float(slopes.min()), "slope_max": float(slopes.max()),
                     "peak_value": float(cp[k]), "peak_at": float(tp[k]),
                     "phi_0.09_over_peak": float(np.interp(0.09, tp, cp) / cp[k]), "phi_at_span": float(cp[-1]),
                     "psi_convex_up_to": float(tp[first_neg]) if first_neg is not None else None,
                     "lip_bound": float(field.lip_bound())}

    # ---- linear regime on a full-size image: x = 0
    gen = torch.Generator(device=dev); gen.manual_seed(0)
    field.set_sigma(args.sigma)
    x0 = torch.zeros(1, 1, 321, 481, device=dev, dtype=torch.float64)
    out["skew_fraction_linear_regime"] = skew_fraction_at(field, x0, args.probes, gen)

    # ---- extreme eigenvalues of Sym J_M at denoising fixed points of full images (shifted power iterations, matrix-free)
    ds = H5ImageDataset(os.path.join(ROOT, "data", "bsd68.h5"))
    rows = []
    for i in range(args.n_images):
        img = ds[i][None].to(dev).double()
        y = img + fixed_noise(img.shape, args.sigma / 255.0, 1234 + i, device=dev).double()
        with torch.no_grad():
            z, info = model.solve(y, sigma=args.sigma)
        field.set_sigma(args.sigma)
        jvp, vjp = make_ops(field, z)
        sym = lambda v: 0.5 * (jvp(v) + vjp(v))  # noqa: E731

        def power(op, iters):
            v = torch.randn(z.shape, device=dev, dtype=torch.float64, generator=gen); v = v / v.norm()
            val = 0.0
            for _ in range(iters):
                w = op(v); val = float((w * v).sum()); v = w / (w.norm() + 1e-30)
            return val, v
        lam_top, _ = power(sym, args.power)  # eigenvalue of Sym J_M of largest magnitude (the positive end)
        r = abs(lam_top) + 1e-6
        _, v = power(lambda u: r * u - sym(u), args.power)  # converges towards the smallest eigenvalue of Sym J_M
        rq = float((v * sym(v)).sum())  # Rayleigh quotient >= lambda_min: a negative value certifies lambda_min < 0
        rows.append({"sym_max_eig": lam_top, "sym_min_eig_upper_bound": rq, "solve_iterations": int(info.iterations.max())})
        print(f"image {i:2d}: Sym J_M max eig {lam_top:.2f}, min eig <= {rq:+.3f} ({int(info.iterations.max())} SAGD iterations)", flush=True)
        torch.cuda.empty_cache()
    out["fixed_points"] = {"n_images": args.n_images, "power_iterations": args.power, "per_image": rows,
                           "sym_min_eig_upper_bound_min": min(r["sym_min_eig_upper_bound"] for r in rows),
                           "sym_min_eig_upper_bound_max": max(r["sym_min_eig_upper_bound"] for r in rows),
                           "sym_max_eig_max": max(r["sym_max_eig"] for r in rows), "negative_on": sum(r["sym_min_eig_upper_bound"] < 0 for r in rows)}
    print(json.dumps({k: v for k, v in out.items() if k != "fixed_points"}, indent=1))
    print("fixed points:", json.dumps({k: v for k, v in out["fixed_points"].items() if k != "per_image"}, indent=1))
    if args.out:
        json.dump(out, open(os.path.join(ROOT, args.out), "w"), indent=1)


if __name__ == "__main__":
    main()
