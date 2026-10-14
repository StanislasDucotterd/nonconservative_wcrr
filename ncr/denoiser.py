"""Deep-equilibrium denoiser: x* = argsolve  x + F(x) = y, trained with implicit differentiation.

Forward:  z* from a batched fixed-point solver (no graph), one differentiable step z = T(z*), and a backward hook
that replaces the incoming gradient g by the solution of (I - J_T^T) v = g (implicit function theorem), where
T(x) = x - alpha (x + F(x) - y) and J_T its Jacobian at z*. This is the scheme of the CRR / WCRR code, generalised
to non-symmetric Jacobians (the adjoint system is solved with Anderson acceleration).
"""
import torch
import torch.nn as nn
from .solvers import solve_fixed_point, solve_adjoint, SolveInfo


class FixedPointDenoiser(nn.Module):
    def __init__(self, field, solver="fista", tol=1e-3, max_iter=1000, step="auto", step_scale=1.0,
                 backward="anderson", bwd_tol=1e-3, bwd_max_iter=50, solver_kw=None, bwd_kw=None, final_step=True,
                 stop="residual"):
        super().__init__()
        self.field = field
        self.solver = solver
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.step = step
        self.step_scale = float(step_scale)
        self.backward = backward
        self.bwd_tol = float(bwd_tol)
        self.bwd_max_iter = int(bwd_max_iter)
        self.solver_kw = dict(solver_kw or {})
        self.bwd_kw = dict(bwd_kw or {})
        self.final_step = bool(final_step)
        self.stop = stop  # 'residual' (||x + F(x) - y|| / ||y|| <= tol) or 'change' (legacy relative change)
        self.last_info = None
        self.last_alpha = None

    def step_size(self):
        """alpha = step_scale / (1 + L) with L the Lipschitz bound of F (the residual G = I + F - y is (1+L)-Lipschitz
        and 1-strongly monotone when F is monotone)."""
        with torch.no_grad():
            L = self.field.lip_bound()
        if self.step == "auto":
            return float(self.step_scale / (1.0 + L))
        return float(self.step)

    def operators(self, y, alpha):
        def G(x, idx):
            return x + self.field(x, idx) - y[idx]

        def T(x, idx):
            return x - alpha * G(x, idx)

        return G, T

    def solve(self, y, solver=None, tol=None, max_iter=None, sigma=None, **kw):
        """Fixed point without any graph (evaluation / diagnostics). sigma: noise level(s) in 0-255 units for
        sigma-conditioned fields (None = the activation's default)."""
        skw = dict(self.solver_kw)
        skw.update(kw)
        self.field.set_sigma(sigma)
        with torch.no_grad(), self.field.cached():
            alpha = self.step_size()
            G, T = self.operators(y, alpha)
            z, info = solve_fixed_point(solver or self.solver, G, T, y, alpha, tol or self.tol, max_iter or self.max_iter,
                                        scale=y.flatten(1).norm(dim=1) + 1e-12, stop=skw.pop("stop", self.stop), **skw)
        self.last_info = info
        self.last_alpha = alpha
        return z, info

    def forward(self, y, sigma=None):
        self.field.set_sigma(sigma)
        with self.field.cached():
            alpha = self.step_size()
            self.last_alpha = alpha
            G, T = self.operators(y, alpha)
            with torch.no_grad():
                z, info = solve_fixed_point(self.solver, G, T, y, alpha, self.tol, self.max_iter,
                                            scale=y.flatten(1).norm(dim=1) + 1e-12, stop=self.stop, **self.solver_kw)
            self.last_info = info
            idx = torch.arange(y.shape[0], device=y.device)
            if not (self.training and torch.is_grad_enabled()):
                return T(z, idx) if self.final_step else z
            z = T(z, idx)  # graph w.r.t. the parameters (z* enters as a constant)
            z0 = z.detach().requires_grad_(True)
            f0 = T(z0, idx)

        def hook(g):
            def vjp(v):
                return torch.autograd.grad(f0, z0, v, retain_graph=True)[0]
            return solve_adjoint(self.backward, vjp, g, tol=self.bwd_tol, max_iter=self.bwd_max_iter, **self.bwd_kw)

        z.register_hook(hook)
        return z

    def stats(self):
        info = self.last_info
        if info is None:
            return {}
        return {
            "fp_iters_mean": float(info.iterations.float().mean()),
            "fp_iters_max": float(info.iterations.max()),
            "fp_residual_max": float(info.residual.max()),
            "fp_converged_frac": float(info.converged.float().mean()),
            "alpha": float(self.last_alpha or 0.0),
        }


def build_denoiser(field, cfg):
    cfg = dict(cfg or {})
    return FixedPointDenoiser(field, **cfg)
