#!/usr/bin/env python
"""Assert-based test suite (no pytest needed):  python tests/run_tests.py [--device cuda:0]"""
import argparse
import os
import sys
import math
import torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ncr.models import RidgeField
from ncr.denoiser import FixedPointDenoiser
from ncr.solvers import anderson, fista
from ncr import diagnostics
from ncr.activations import LinearSplineActivation

RESULTS = []
SPLINE = {"type": "spline", "n_knots": 41, "span": 2.0, "compile": False}


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond), detail))
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))


def random_field(dev, untied=None, channels=(1, 4, 8, 12), **act):
    f = RidgeField(channels=list(channels), activation={**SPLINE, **act}, untied=untied).to(dev).double()
    with torch.no_grad():
        f.act.theta.copy_(3 * torch.randn_like(f.act.theta))
        f.act.log_mu.fill_(1.0)
        if f.untied:
            f.delta.add_(0.3 * torch.randn_like(f.delta))
    return f


def test_diagnostics(dev):
    torch.manual_seed(0)
    x = torch.rand(1, 1, 32, 32, device=dev, dtype=torch.float64)
    f = random_field(dev)
    asym = diagnostics.asymmetry_index(f, x, n_probes=3)
    check("tied field (a gradient) has zero Jacobian asymmetry", asym < 1e-8, f"{asym:.1e}")
    lip = diagnostics.lipschitz_estimate(f, x, iters=40)
    check("Lipschitz estimate <= bound", lip <= float(f.lip_bound()) * (1 + 1e-6), f"{lip:.3f} <= {float(f.lip_bound()):.3f}")
    fu = random_field(dev, untied="delta_last")
    asym_u = diagnostics.asymmetry_index(fu, x, n_probes=3)
    check("untied field has non-zero asymmetry", asym_u > 1e-2, f"{asym_u:.3f}")


def test_solvers_and_implicit_gradient(dev):
    torch.manual_seed(0)
    torch.set_default_dtype(torch.float64)
    try:
        n = 60
        Q, _ = torch.linalg.qr(torch.randn(n, n))
        M = 0.9 * Q @ torch.diag(torch.rand(n)) @ Q.T
        S = torch.randn(n, n); M = M + 0.2 * (S - S.T) / math.sqrt(n)
        b = torch.randn(2, n)
        T = lambda x, idx: (x.view(len(idx), -1) @ M.T + b[idx]).view(len(idx), 1, 1, n)  # noqa: E731
        xs = torch.linalg.solve(torch.eye(n) - M, b.T).T
        xa, ia = anderson(T, torch.zeros(2, 1, 1, n), tol=1e-12, max_iter=500)
        check("Anderson solves a non-symmetric linear fixed point", (xa.view(2, n) - xs).norm() / xs.norm() < 1e-8, f"iters {ia.iterations.max():.0f}")
        # SAGD on a symmetric positive definite problem: G(x) = A x - c, alpha = 1 / ||A||
        A = Q @ torch.diag(0.5 + torch.rand(n)) @ Q.T; c = torch.randn(1, n)
        Gl = lambda x, idx: (x.view(len(idx), -1) @ A.T - c[idx]).view(len(idx), 1, 1, n)  # noqa: E731
        xg, ig = fista(Gl, torch.zeros(1, 1, 1, n), 1.0 / float(torch.linalg.matrix_norm(A, 2)), tol=1e-12, max_iter=5000, stop="change")
        ref = torch.linalg.solve(A, c.T).T
        check("SAGD (accelerated gradient with restart) solves a convex problem", (xg.view(1, n) - ref).norm() / ref.norm() < 1e-8, f"iters {ig.iterations.max():.0f}")
        # implicit gradient vs unrolled autograd on tiny fields (mu = 0.8 so that Lip(M) < 1 and the unrolled reference converges)
        for untied in (None, "delta_last"):
            f = RidgeField(channels=(1, 4, 8), kernel_sizes=(3, 3), activation={**SPLINE, "n_knots": 9, "mu_init": 0.8}, untied=untied).to(dev)
            with torch.no_grad():
                f.act.theta.copy_(torch.randn_like(f.act.theta))
                if f.untied:
                    f.delta.add_(0.05 * torch.randn_like(f.delta))
            y = torch.rand(1, 1, 10, 10, device=dev)
            tgt = torch.rand_like(y)
            den = FixedPointDenoiser(f, solver="anderson", tol=1e-12, max_iter=500, backward="anderson", bwd_tol=1e-12, bwd_max_iter=400).to(dev).train()
            loss = ((den(y) - tgt) ** 2).sum()
            grads_impl = torch.autograd.grad(loss, [p for p in f.parameters() if p.requires_grad])
            alpha = den.step_size()
            x = y.clone()
            with f.cached():
                for _ in range(1500):
                    x = x - alpha * (x + f(x) - y)
            loss_u = ((x - tgt) ** 2).sum()
            grads_unr = torch.autograd.grad(loss_u, [p for p in f.parameters() if p.requires_grad])
            num = sum(((a - b_) ** 2).sum() for a, b_ in zip(grads_impl, grads_unr)).sqrt()
            den_ = sum((b_ ** 2).sum() for b_ in grads_unr).sqrt()
            rel = (num / den_).item()
            check(f"implicit gradient matches the unrolled gradient ({'untied' if f.untied else 'tied'})", rel < 1e-4, f"rel err {rel:.1e}")
    finally:
        torch.set_default_dtype(torch.float32)


def test_linear_spline(dev):
    torch.manual_seed(0)
    G = 5
    dd = dict(device=dev, dtype=torch.float64)
    act = LinearSplineActivation(G, 1, n_knots=41, span=2.0, compile=False).to(dev).double()
    u = (0.01 * torch.randn(3, G, 1, 8, 8, **dd)).requires_grad_(True)
    check("spline: zero at init (output and Lipschitz bound)", act(u).abs().max().item() == 0.0 and float(act.max_block_norm().max()) == 0.0)
    with torch.no_grad():
        act.theta.copy_(3 * torch.randn_like(act.theta))  # most slopes saturate at +-1
        act.log_alpha.copy_(2 + torch.randn_like(act.log_alpha)); act.log_mu.fill_(1.5)
    sl = act.slopes()
    check("spline: slopes in [-1, 1], phi(0) = 0", bool(sl.min() >= -1) and bool(sl.max() <= 1) and float(act.coefficients()[:, act.mid].abs().max()) == 0.0)
    g = torch.autograd.grad(act.potential(u).sum(), u)[0]
    err = (g - act(u)).abs().max().item()
    check("spline = gradient of its potential (inside the knots)", err < 1e-10, f"{err:.1e}")
    a, o, lam = act._scales()
    u2 = (torch.randn(2, G, 1, 4, 4, **dd) * 5 / a.view(1, G, 1, 1, 1)).requires_grad_(True)  # mostly beyond the knot range
    g2 = torch.autograd.grad(act.potential(u2).sum(), u2)[0]
    err2 = (g2 - act(u2)).abs().max().item()
    check("spline = gradient of its potential (beyond the knots, constant continuation)", err2 < 1e-10, f"{err2:.1e}")
    raw = LinearSplineActivation(G, 1, n_knots=41, span=2.0, compile=False, learn_scales=True).to(dev).double()  # alpha = 1, mu = 1
    with torch.no_grad():
        raw.theta.copy_(act.theta); raw.log_mu.zero_()
    ref = lam * raw(u.detach() * a) / a
    err = (act(u) - ref).abs().max().item()
    check("spline(crr): Phi = mu * phi(alpha u) / alpha", err < 1e-12, f"{err:.1e}")
    errL = (act.max_block_norm() - lam * act.slopes().abs().max(dim=1).values).abs().max().item()
    check("spline(crr): Lipschitz bound = mu * max|slope| (alpha-free)", errL < 1e-12, f"{errL:.1e}")
    v = torch.randn_like(u)
    jv = torch.func.jvp(act, (u.detach(),), (v,))[1]
    with torch.no_grad():  # exact Jacobian: mu * slope of the cell containing alpha u (0 beyond the knots)
        t = act.mid + u.detach() * a / act.h
        inside = (t >= 0) & (t <= act.K - 1)
        cell = t.clamp(0, act.K - 2).floor().long()
        slope = torch.gather(act.slopes().view(1, G, 1, 1, -1).expand(u.shape[0], G, 1, u.shape[3], -1), 4, cell) * inside
        ref = lam * slope * v
    check("spline: forward-mode AD == exact per-cell slope", (jv - ref).abs().max().item() < 1e-10, f"{(jv - ref).abs().max().item():.1e}")
    gu = torch.autograd.grad((act(u) ** 2).sum(), u, create_graph=True)[0]
    gt = torch.autograd.grad(gu.sum(), act.theta, allow_unused=True)[0]
    check("spline: double backward reaches the spline parameters", gt is not None and bool(torch.isfinite(gt).all()) and float(gt.abs().sum()) > 0)
    if dev.startswith("cuda"):
        act_c = LinearSplineActivation(G, 1, n_knots=41, span=2.0, compile=True).to(dev)
        with torch.no_grad():
            act_c.theta.copy_(act.theta.float()); act_c.log_alpha.copy_(act.log_alpha.float()); act_c.log_mu.copy_(act.log_mu.float())
            xb = 0.02 * torch.randn(64, G, 1, 64, 64, device=dev)
            ab = act_c(xb)  # compiled (no graph)
        bb = act_c(xb.clone().requires_grad_(True)).detach()  # eager (graph)
        check("spline: compiled no-grad path == eager path at the training shape", (ab - bb).abs().max().item() < 1e-5, f"{(ab - bb).abs().max().item():.1e}")
    for par in ("slopes", "coefficients"):
        acts = LinearSplineActivation(G, 1, n_knots=41, span=2.0, compile=False, antisymmetric=True, parametrization=par).to(dev).double()
        with torch.no_grad():
            acts.theta.copy_(3 * torch.randn_like(acts.theta))
        uu = (torch.randn(2, G, 1, 6, 6, **dd) * 3 / torch.exp(acts.log_alpha)).detach()
        odd = (acts(uu) + acts(-uu)).abs().max().item()
        s = acts.slopes()
        check(f"spline(antisymmetric, {par}): Phi(-u) = -Phi(u), slopes mirror-symmetric", odd < 1e-12 and (s - s.flip(1)).abs().max().item() < 1e-12
              and s.shape[1] == acts.K - 1 and (par != "slopes" or acts.theta.shape[1] == (acts.K - 1) // 2), f"{odd:.1e}")
    # field level: certified weak-monotonicity modulus and Lipschitz bound against the exact Jacobian on a small image
    for mode in (None, "delta_last"):
        f = random_field(dev, untied=mode)
        x = torch.rand(1, 1, 16, 16, **dd)
        J = torch.autograd.functional.jacobian(lambda z: f(z.view(1, 1, 16, 16)).flatten(), x.flatten())
        lam_min = float(torch.linalg.eigvalsh(0.5 * (J + J.T)).min())
        rho = float(f.weak_monotone_bound())
        L = float(f.lip_bound())
        smax = float(torch.linalg.matrix_norm(J, 2))
        tag = "untied" if f.untied else "tied"
        check(f"field({tag}): lambda_min(Sym J) >= -rho_certified", lam_min >= -rho * (1 + 1e-6) - 1e-9, f"lam_min {lam_min:+.4f} rho {rho:.4f}")
        check(f"field({tag}): ||J|| <= Lipschitz bound", smax <= L * (1 + 1e-6), f"{smax:.4f} <= {L:.4f}")
        if not f.untied:
            check("field(tied): symmetric Jacobian", float((J - J.T).abs().max()) < 1e-10)
    # sigma-conditioned scales (the WCRR mechanism): alpha_c(sigma) = exp(s_c(sigma)) / (sigma + eps), sigma per sample
    actS = LinearSplineActivation(G, 1, n_knots=41, span=2.0, compile=False, scales="crr_sigma", sigma_spline_init=5.0).to(dev).double()
    with torch.no_grad():
        actS.theta.copy_(3 * torch.randn_like(actS.theta))
    sig = torch.tensor([5.0, 25.0], **dd)
    a, o, lam = actS._scales(sig)
    expect = torch.tensor([math.exp(5.0) / (5.0 + 1e-5), math.exp(5.0) / (25.0 + 1e-5)], **dd)
    check("spline(crr_sigma): alpha = exp(s(sigma)) / (sigma + eps) at init, per sample", torch.allclose(a.view(2, G), expect.view(2, 1).expand(2, G)) and a.shape == (2, G, 1, 1, 1))
    ub = torch.randn(2, G, 1, 4, 4, **dd) * 0.05
    yb = actS(ub, sigma=sig)
    ysep = torch.cat([actS(ub[:1], sigma=sig[:1]), actS(ub[1:], sigma=sig[1:])])
    check("spline(crr_sigma): batched per-sample sigma == separate evaluations", float((yb - ysep).abs().max()) < 1e-12)
    ubg = ub.clone().requires_grad_(True)
    gS = torch.autograd.grad(actS.potential(ubg, sigma=sig).sum(), ubg)[0]
    check("spline(crr_sigma): gradient of the potential", float((gS - yb).abs().max()) < 1e-10, f"{float((gS - yb).abs().max()):.1e}")
    check("spline(crr_sigma): Lipschitz gain independent of sigma (= mu)", float((actS.gain() - lam).abs().max()) < 1e-12)
    fS = RidgeField(channels=[1, 4, 8, 6], untied="delta_last", activation={**SPLINE, "n_knots": 9, "scales": "crr_sigma", "mu_init": 0.8}).to(dev).double()
    with torch.no_grad():
        fS.act.theta.copy_(torch.randn_like(fS.act.theta)); fS.delta.add_(0.05 * torch.randn_like(fS.delta))
    den = FixedPointDenoiser(fS, solver="agd", stop="change", tol=1e-10, max_iter=2000, backward="anderson").to(dev).eval()
    yy = torch.rand(3, 1, 12, 12, **dd)
    ss = torch.tensor([2.0, 10.0, 30.0], **dd)
    with torch.no_grad():
        zb, _ = den.solve(yy, sigma=ss)
        zs = torch.cat([den.solve(yy[i:i + 1], sigma=ss[i:i + 1])[0] for i in range(3)])
        z_def, _ = den.solve(yy[:1])
        z_25, _ = den.solve(yy[:1], sigma=torch.tensor([25.0], **dd))
    check("denoiser (SAGD): batched solve with per-sample sigma == individual solves", float((zb - zs).abs().max()) < 1e-7, f"{float((zb - zs).abs().max()):.1e}")
    check("denoiser: no sigma given == default sigma 25", float((z_def - z_25).abs().max()) < 1e-12)
    sh = LinearSplineActivation(G, 1, n_knots=41, span=2.0, compile=False, scales="crr_sigma", shared=True).to(dev).double()
    with torch.no_grad():
        sh.theta.copy_(3 * torch.randn_like(sh.theta)); sh.sigma_spline.add_(torch.randn_like(sh.sigma_spline))
    per = LinearSplineActivation(G, 1, n_knots=41, span=2.0, compile=False, scales="crr_sigma").to(dev).double()
    with torch.no_grad():
        per.theta.copy_(sh.theta.expand(G, -1)); per.sigma_spline.copy_(sh.sigma_spline)
    ub2 = torch.randn(2, G, 1, 4, 4, **dd) * 0.05
    check("spline(shared): == per-channel splines with identical rows; parameters / G",
          float((sh(ub2, sigma=sig) - per(ub2, sigma=sig)).abs().max()) < 1e-12 and sh.theta.shape[0] == 1 and sh.theta.numel() * G == per.theta.numel())
    tv2 = actS.penalties()["tv2"]
    g_theta, g_sigma = torch.autograd.grad(tv2, [actS.theta, actS.sigma_spline], allow_unused=True)
    check("spline(crr_sigma): TV2 penalty has no gradient w.r.t. the sigma spline", g_sigma is None and g_theta is not None and float(g_theta.abs().sum()) > 0)


def test_anderson_robustness(dev):
    """The Anderson solver must never raise and never return non-finite iterates: a sample that blows up is frozen at
    its last finite pair (converged=False, infinite residual) and a singular least-squares system falls back to a
    pseudo-inverse solution."""
    dd = dict(device=dev, dtype=torch.float64)
    torch.manual_seed(0)
    A = torch.randn(10, 10, **dd); A = 0.6 * A / torch.linalg.matrix_norm(A, 2)
    b = torch.randn(4, 10, **dd)
    ref = torch.linalg.solve(torch.eye(10, **dd) - A, b.T).T
    calls = {"n": 0}

    def T(x, idx):
        calls["n"] += 1
        out = x @ A.T + b[idx]
        if calls["n"] > 5:
            out = out.clone(); out[idx == 2] = float("nan")
        return out
    x0 = torch.zeros(4, 10, **dd)
    x, info = anderson(T, x0, m=5, tol=1e-10, max_iter=200, stop="change")
    err = float((x[[0, 1, 3]] - ref[[0, 1, 3]]).abs().max())
    check("anderson: diverging sample frozen, others converge, output finite",
          bool(torch.isfinite(x).all()) and info.converged.tolist() == [True, True, False, True] and bool(torch.isinf(info.residual[2])) and err < 1e-9,
          f"converged {info.converged.tolist()} err {err:.1e}")
    orig = torch.linalg.solve_ex; flagged = {"n": 0}

    def fake(Hm, r):
        sol, inf = orig(Hm, r)
        if flagged["n"] == 0:
            flagged["n"] = 1
            return torch.full_like(sol, float("nan")), torch.ones_like(inf)
        return sol, inf
    torch.linalg.solve_ex = fake
    try:
        x2, info2 = anderson(lambda x, idx: x @ A.T + b[idx], x0, m=5, tol=1e-10, max_iter=200, stop="change")
    finally:
        torch.linalg.solve_ex = orig
    err2 = float((x2 - ref).abs().max())
    check("anderson: singular least-squares system falls back to the pseudo-inverse", flagged["n"] == 1 and bool(info2.converged.all()) and err2 < 1e-9, f"err {err2:.1e}")


def test_data_and_trainer(dev):
    from ncr.data import SyntheticImageDataset, H5ImageDataset
    ds = SyntheticImageDataset(4, 32)
    x = ds[0]
    check("synthetic dataset shape/range", x.shape == (1, 32, 32) and 0 <= x.min() and x.max() <= 1)
    p = os.path.join(ROOT, "data", "bsd68.h5")
    if os.path.exists(p):
        d = H5ImageDataset(p, patch_size=40, augment=True)
        check("bsd68.h5 loads (patch)", d[0].shape == (1, 40, 40) and len(d) == 68)
    from ncr.utils import load_config
    from ncr.trainer import Trainer
    cfg = load_config(os.path.join(ROOT, "configs", "nonconservative_wcrr.yaml"))
    cfg.update({"exp_name": "_test_smoke", "log_dir": os.path.join(ROOT, "exps")})
    cfg["data"].update({"synthetic": True, "synthetic_n": 8, "synthetic_val_n": 2, "synthetic_val_size": 48, "patch_size": 32, "batch_size": 4, "num_workers": 0, "val_every": 1})
    cfg["model"].update({"channels": [1, 4, 8, 16]})
    cfg["train"].update({"epochs": 2, "tensorboard": False})
    tr = Trainer(cfg, dev)
    tr.train()
    check("trainer smoke run (the paper's config on synthetic data, SAGD forward, Anderson backward)", os.path.exists(os.path.join(tr.exp_dir, "last.pth")) and tr.step == 4)
    import shutil
    shutil.rmtree(tr.exp_dir, ignore_errors=True)


def test_checkpoint(dev):
    """The shipped model reproduces its stored per-image PSNRs on the first BSD68 images (needs data/bsd68.h5)."""
    import json
    from ncr.trainer import load_checkpoint
    from ncr.data import H5ImageDataset, fixed_noise
    from ncr.utils import psnr
    ck = os.path.join(ROOT, "trained_models", "nonconservative_wcrr", "last.pth")
    data = os.path.join(ROOT, "data", "bsd68.h5")
    res = os.path.join(ROOT, "results", "bsd68", "nonconservative_wcrr.json")
    if not (os.path.exists(ck) and os.path.exists(data) and os.path.exists(res)):
        print("skip: checkpoint, data or stored results missing")
        return
    field, model, cfg = load_checkpoint(ck, dev)
    stored = json.load(open(res))["psnr"]["25"]
    ds = H5ImageDataset(data)
    vals = []
    with torch.no_grad():
        for i in range(3):
            x = ds[i][None].to(dev)
            y = x + fixed_noise(x.shape, 25 / 255.0, 1234 + i, device=dev)
            vals.append(float(psnr(model(y, sigma=torch.full((1,), 25.0, device=dev)).clamp(0, 1), x).mean()))
    err = max(abs(v - s) for v, s in zip(vals, stored[:3]))
    check("shipped checkpoint reproduces the stored BSD68 PSNRs (sigma 25, first 3 images)", err < 0.02, f"max diff {err:.3f} dB; {[round(v, 3) for v in vals]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = args.device
    for t in [test_diagnostics, test_solvers_and_implicit_gradient, test_linear_spline, test_anderson_robustness, test_data_and_trainer, test_checkpoint]:
        try:
            t(dev)
        except Exception as e:
            import traceback; traceback.print_exc()
            check(f"{t.__name__} raised {type(e).__name__}: {e}", False)
    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed")
    sys.exit(1 if n_fail else 0)
