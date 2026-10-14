#!/usr/bin/env python
"""Sparse-view CT (LoDoPaB-CT, 362x362 slices, parallel beam, 256 detectors, Gaussian noise 0.1 on the sinogram) with the
trained field as the regularizer, under the protocol of the MFoE paper:
  x = zero of  H^T(H x - y) + lambda M_sigma(x),  found by SAGD (accelerated gradient descent with adaptive restart) from the
  FBP, stopped on the relative change of the iterates (1e-5) or after max_iter iterations;
  (lambda, sigma) tuned on the 10 validation slices with the MFoE zeroth-order search, evaluated on the 50 test slices
  (PSNR / SSIM with torchmetrics, as in the papers).
Operators (ODL + ASTRA), the search routine and the data come from the MFoE repository (https://github.com/StanislasDucotterd/MFoE):
clone it, uncompress inverse_problems/ct/val_data.tar.gz and test_data.tar.gz and run inverse_problems/ct/get_data.py, then
    python scripts/ct_reconstruction.py --mfoe-dir /path/to/MFoE --angles 60 --device cuda:0
Results: results/ct/<name>_<angles>/results.json (tuned lambda and sigma, per-slice PSNR / SSIM) and the reconstructions.
"""
import argparse
import json
import os
import sys
import time
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ncr.trainer import load_checkpoint  # noqa: E402
from ncr.solvers import solve_fixed_point  # noqa: E402


def mfoe_imports(mfoe_dir):
    """get_operators (ODL + ASTRA) and tune_hyperparameters from the MFoE repository; returns them with the CT data folder."""
    ct_dir = os.path.join(os.path.expanduser(mfoe_dir), "inverse_problems", "ct")
    for p in (ct_dir, os.path.expanduser(mfoe_dir)):
        sys.path.insert(0, p)
    from ct_forward_utils import get_operators  # noqa: E402
    from inverse_problems.tune_hyperparameters import tune_hyperparameters  # noqa: E402
    return get_operators, tune_hyperparameters, ct_dir


def load_split(ct_dir, split):
    folder = os.path.join(ct_dir, split)
    names = sorted(os.listdir(folder), key=lambda s: int(s.replace("image", "")))
    return [(n, os.path.join(folder, n)) for n in names]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="trained_models/nonconservative_wcrr/last.pth")
    ap.add_argument("--name", default="ours", help="results go to results/ct/<name>_<angles>/")
    ap.add_argument("--mfoe-dir", default=os.path.expanduser("~/MFoE"), help="clone of the MFoE repository (operators, search, data)")
    ap.add_argument("--angles", type=int, default=60, choices=[20, 40, 60])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tol", type=float, default=1e-5, help="relative change of the iterates")
    ap.add_argument("--max-iter", type=int, default=1000)
    ap.add_argument("--lamb-init", type=float, default=30.0, help="start of the zeroth-order search")
    ap.add_argument("--sigma-init", type=float, default=10.0, help="start of the zeroth-order search (0-255 units)")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "results", "ct"))
    args = ap.parse_args()
    from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
    get_operators, tune_hyperparameters, ct_dir = mfoe_imports(args.mfoe_dir)
    dev = args.device
    torch.cuda.set_device(dev)
    fwd, fbp, bp = get_operators(img_size=362, space_range=1, num_angles=args.angles, det_shape=256, device=dev, fix_scaling=True)
    with torch.no_grad():  # ||H^T H||_2 by power iteration (the data term's Lipschitz constant)
        u = torch.randn(1, 1, 362, 362, device=dev)
        for _ in range(60):
            u = bp(fwd(u)); data_lip = float(u.norm()); u = u / data_lip
    field = load_checkpoint(os.path.join(ROOT, args.checkpoint), dev)[0].eval()
    L = float(field.lip_bound())  # sigma-independent (the scaling only moves the knots)
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(dev)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(dev)
    print(f"{args.name} ({args.checkpoint}), {args.angles} angles: ||H^T H|| = {data_lip:.1f}, Lip(M) = {L:.1f}, solver agd / change {args.tol:g} / cap {args.max_iter}", flush=True)
    out_dir = os.path.join(args.out_dir, f"{args.name}_{args.angles}")
    os.makedirs(out_dir, exist_ok=True)

    def reconstruct(y, lamb, sigma):
        field.set_sigma(float(sigma))
        alpha = 1.0 / (data_lip + float(lamb) * L)
        G = lambda x, idx: bp(fwd(x) - y[idx]) + float(lamb) * field(x)  # noqa: E731
        T = lambda x, idx: x - alpha * G(x, idx)  # noqa: E731
        x0 = fbp(y)
        with torch.no_grad():
            x, info = solve_fixed_point("agd", G, T, x0, alpha, tol=args.tol, max_iter=args.max_iter, scale=y.flatten(1).norm(dim=1), stop="change")
        return x, int(info.iterations.max())

    def evaluate(split, lamb, sigma, save=False):
        ps, ss, its, times = [], [], [], []
        for name, folder in load_split(ct_dir, split):
            x_gt = torch.load(os.path.join(folder, "image.pth"), weights_only=False).to(dev)
            y = torch.load(os.path.join(folder, f"y_{args.angles}_256.pth"), weights_only=False).to(dev)
            t0 = time.time()
            x, it = reconstruct(y, lamb, sigma)
            torch.cuda.synchronize(dev)
            times.append(time.time() - t0); its.append(it)
            xc = x.clamp(0, 1)
            ps.append(float(psnr(xc, x_gt))); ss.append(float(ssim(xc, x_gt)))
            if save:
                torch.save(x.cpu(), os.path.join(out_dir, f"pred_{name}.pt"))
        return dict(psnr=sum(ps) / len(ps), ssim=sum(ss) / len(ss), iters_mean=sum(its) / len(its), iters_max=max(its),
                    cap_hits=sum(1 for i in its if i >= args.max_iter), time_mean=sum(times) / len(times), psnr_per_image=ps, ssim_per_image=ss, iters=its)

    log = []

    def test_hyperparameter(lamb, sigma):
        r = evaluate("val_data", float(lamb), float(sigma))
        log.append(dict(lamb=float(lamb), sigma=float(sigma), **{k: v for k, v in r.items() if not isinstance(v, list)}))
        print(f"   lambda {float(lamb):.5g}, sigma {float(sigma):.4g}: val PSNR {r['psnr']:.3f} dB, iterations {r['iters_mean']:.0f} (max {r['iters_max']}, cap hits {r['cap_hits']}), {r['time_mean']:.1f} s/img", flush=True)
        return r["psnr"]

    t0 = time.time()
    lamb_init = args.lamb_init
    best_lamb, best_sigma = tune_hyperparameters(test_hyperparameter, torch.tensor(lamb_init), torch.tensor(args.sigma_init))
    best_lamb, best_sigma = float(best_lamb), float(best_sigma)
    print(f"tuning done in {(time.time()-t0)/60:.1f} min: lambda = {best_lamb:.5g}, sigma = {best_sigma:.4g} ({len(log)} evaluations)", flush=True)
    test = evaluate("test_data", best_lamb, best_sigma, save=True)
    print(f"TEST {args.name} {args.angles} angles: PSNR {test['psnr']:.3f} dB, SSIM {test['ssim']:.4f}, iterations {test['iters_mean']:.0f} (max {test['iters_max']}, cap hits {test['cap_hits']}/50), {test['time_mean']:.1f} s/img", flush=True)
    json.dump(dict(model=args.name, checkpoint=args.checkpoint, angles=args.angles, solver=dict(method="agd", stop="change", tol=args.tol, max_iter=args.max_iter),
                   data_lip=data_lip, lip_F=L, best_lamb=best_lamb, best_sigma=best_sigma, tuning=log, test=test),
              open(os.path.join(out_dir, "results.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
