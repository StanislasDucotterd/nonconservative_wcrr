"""Jacobian diagnostics of a field F (or any map) at given points, using only matrix-vector products.

asymmetry_index     ||J - J^T||_F / ||J||_F estimated with Gaussian probes (0 for conservative fields)
sym_extreme_eigs    largest and smallest eigenvalues of Sym(J) = (J + J^T)/2 (power iterations); monotone iff min >= 0
lipschitz_estimate  ||J||_2 by power iteration on J^T J (lower bound on Lip(F), to compare with field.lip_bound())
"""
import torch


def _jvp(fn, x, v):
    try:
        return torch.func.jvp(fn, (x,), (v,))[1]
    except Exception:  # fallback: double-backward trick
        return torch.autograd.functional.jvp(fn, x, v, create_graph=False)[1]


def _vjp(fn, x, v):
    xx = x.detach().requires_grad_(True)
    with torch.enable_grad():
        y = fn(xx)
        return torch.autograd.grad(y, xx, v)[0]


def asymmetry_index(fn, x, n_probes=8, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    num, den = 0.0, 0.0
    for _ in range(n_probes):
        v = torch.randn(x.shape, generator=g).to(x.device)
        jv = _jvp(fn, x, v)
        jtv = _vjp(fn, x, v)
        num += float(((jv - jtv) ** 2).sum())
        den += float((jv ** 2).sum())
    return (num / max(den, 1e-30)) ** 0.5


def _sym_apply(fn, x, v):
    return 0.5 * (_jvp(fn, x, v) + _vjp(fn, x, v))


def sym_extreme_eigs(fn, x, iters=60, seed=0):
    """(lam_max, lam_min) of the symmetric part of J at x by shifted power iterations (single sample x).
    Step 1: spectral radius r = largest |eigenvalue|; step 2: lam_max = top eigenvalue of (Sym + r I) - r;
    step 3: lam_min = r - top eigenvalue of (r I - Sym)."""
    g = torch.Generator(device="cpu").manual_seed(seed)

    def power(op):
        v = torch.randn(x.shape, generator=g).to(x.device, x.dtype)
        v = v / v.norm()
        val = 0.0
        for _ in range(iters):
            w = op(v)
            val = float((w * v).sum())
            v = w / (w.norm() + 1e-30)
        return val

    r = abs(power(lambda v: _sym_apply(fn, x, v))) + 1e-6
    lam_max = power(lambda v: _sym_apply(fn, x, v) + r * v) - r
    lam_min = r - power(lambda v: r * v - _sym_apply(fn, x, v))
    return lam_max, lam_min


def lipschitz_estimate(fn, x, iters=60, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    v = torch.randn(x.shape, generator=g).to(x.device)
    v = v / v.norm()
    s = 0.0
    for _ in range(iters):
        w = _vjp(fn, x, _jvp(fn, x, v))
        s = float((w * v).sum()) ** 0.5
        v = w / (w.norm() + 1e-30)
    return s


def field_report(field, x, n_probes=4, iters=40):
    """All diagnostics at a single point x (1,1,H,W)."""
    fn = lambda z: field(z)
    lam_max, lam_min = sym_extreme_eigs(fn, x, iters=iters)
    return {
        "asymmetry": asymmetry_index(fn, x, n_probes=n_probes),
        "sym_eig_max": lam_max,
        "sym_eig_min": lam_min,
        "lip_estimate": lipschitz_estimate(fn, x, iters=iters),
        "lip_bound": float(field.lip_bound()),
    }


def convergence_curves(denoiser, y, methods=("fista", "picard", "anderson", "extragradient"), max_iter=300, tol=1e-8):
    """Residual history of each solver on the same batch (for plots / tables)."""
    out = {}
    for m in methods:
        try:
            _, info = denoiser.solve(y, solver=m, tol=tol, max_iter=max_iter)
            out[m] = info.history
        except Exception as e:  # pragma: no cover
            out[m] = f"failed: {e}"
    return out
