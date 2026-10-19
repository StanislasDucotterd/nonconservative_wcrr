"""Group activations Phi acting on channel groups: input/output shape (B, G, k, H, W), G groups of k channels.
This repository ships the univariate learnable linear spline of CRR / WCRR (LinearSplineActivation), applied componentwise
(k = 1 in the paper), with the WCRR noise-level scaling."""
import math
import torch
import torch.nn as nn


class GroupActivation(nn.Module):
    is_conservative = False

    def __init__(self, groups, group_size):
        super().__init__()
        self.G, self.k = int(groups), int(group_size)

    def forward(self, u):
        raise NotImplementedError

    def potential(self, u):
        """(B,) potential whose gradient (w.r.t. u) is Phi(u); None if not conservative."""
        return None

    def max_block_norm(self):
        """(G,) upper bounds on sup_u ||J_{Phi_g}(u)||_2 (used in the Lipschitz bound of the field)."""
        raise NotImplementedError

    def block_eig_range(self):
        """(lo, hi), each (G,): bounds on the eigenvalues of Sym J_{Phi_g}(u) over all u (lo <= 0 <= hi); used for the
        certified weak-monotonicity modulus of the field. Default: symmetric bound; monotone activations give lo = 0."""
        hi = self.max_block_norm()
        return -hi, hi

    def penalties(self):
        """dict of scalar differentiable penalties (empty if none)."""
        return {}

    def certificate(self, rho=0.0):
        """dict describing what is provable: monotone (bool or None), conservative (bool), margins..."""
        return {"monotone": None, "conservative": self.is_conservative}


def _spline_lookup(u, mu_over_h, mid_t, table, offsets, K):
    """Piecewise-linear interpolation of the per-group tables `table` (flattened (G*K,)) at the continuous knot
    indices mid + u * mu_over_h, clamped to the knot range (constant continuation). Plain tensor ops: autograd
    gives the scatter-add into the table and the slope for the input; forward-mode AD and double backward work."""
    t = torch.addcmul(mid_t, u, mu_over_h).clamp(0, K - 1)
    idx = t.detach().int().clamp_(max=K - 2)
    frac = t - idx
    base = idx.add_(offsets).view(-1)
    return torch.lerp(table.index_select(0, base).view_as(t), table[1:].index_select(0, base).view_as(t), frac)


_COMPILED = {"fn": None, "ok": True}

# torch.compile is used ONLY when no graph is recorded. Do not compile the differentiable path: on 2026-10-05 the inductor
# backward of the (equally elementwise) MFoE projection was found to be wrong at the training shape (64, 15, 4, 64, 64)
# while correct at (8, 15, 4, 32, 32) and while the compiled forward values were exact at both shapes (torch 2.5.1).


def spline_lookup(u, mu_over_h, mid_t, table, offsets, K, compiled=False):
    """Fused (torch.compile, dynamic shapes) when no graph is being recorded, eager autograd otherwise."""
    if compiled and _COMPILED["ok"] and not torch.is_grad_enabled():
        try:
            if _COMPILED["fn"] is None:
                _COMPILED["fn"] = torch.compile(_spline_lookup, dynamic=True)
            return _COMPILED["fn"](u, mu_over_h, mid_t, table, offsets, K)
        except Exception as e:  # torch.compile unavailable on this platform: fall back to eager for good
            _COMPILED["ok"] = False
            print(f"[LinearSplineActivation] torch.compile failed ({type(e).__name__}: {str(e)[:100]}); eager mode")
    return _spline_lookup(u, mu_over_h, mid_t, table, offsets, K)


class LinearSplineActivation(GroupActivation):
    """Learnable univariate linear spline (CRR / WCRR style), applied componentwise:
        Phi_c(u) = mu * phi_c(alpha_c u) / alpha_c,
    phi_c continuous piecewise-linear on K uniform knots of [-span, span] (K odd: 0 is a knot), constant outside,
    parametrised by its K-1 slopes clamped to [slope_min, slope_max] (default [-1, 1]: 1-Lipschitz, NOT forced to be
    increasing), anchored at phi_c(0) = 0 and ZERO at init. shared=True: one spline phi for every channel (WCRR's layout),
    the scales stay per channel. antisymmetric=True: odd phi (even potential), only the right half of the slopes is learned.
    Scales:
      scales='crr'       alpha_c per channel and one global strength mu (WCRR's mechanism at a fixed noise level), both
                         log-parametrised: Phi_c(u) = mu * phi_c(alpha_c u) / alpha_c.
      scales='crr_sigma' the full WCRR mechanism: alpha_c(sigma) = exp(s_c(sigma)) / (sigma + 1e-5) with s_c a learnable
                         linear spline of the noise level sigma (0-255 units; sigma_knots knots on sigma_range, initialised
                         at the constant sigma_spline_init), mu global. sigma is given per sample to forward / potential
                         (shape (B,)) or taken from `default_sigma`.
    alpha only moves the knots: the effective slopes are mu * phi_c' for every channel and every sigma, so the Lipschitz
    and weak-convexity bounds do not depend on alpha or sigma.  J_{Phi_c}(u) = mu * phi_c'(alpha_c u).
    Conservative: Phi_c = psi_c' with psi_c(u) = mu / alpha_c * int_0^{alpha_c u} phi_c."""
    is_conservative = True

    def __init__(self, groups, group_size=1, n_knots=201, span=4.0, slope_min=-1.0, slope_max=1.0, scales="crr",
                 alpha_init=1.0, mu_init=math.exp(4.0), mu_max=math.exp(10.0), learn_scales=True,
                 parametrization="slopes", antisymmetric=False, sigma_knots=11, sigma_range=(0.0, 30.0),
                 sigma_spline_init=5.0, default_sigma=25.0, shared=False, compile=True, lambda_init=None, lambda_max=None):
        super().__init__(groups, group_size)
        if lambda_init is not None:  # legacy names of the strength (called lambda before the paper's notation changed to mu)
            mu_init = lambda_init
        if lambda_max is not None:
            mu_max = lambda_max
        K = int(n_knots) + (1 - int(n_knots) % 2)  # odd number of knots so that 0 is a knot
        self.K, self.span, self.h, self.mid = K, float(span), 2 * float(span) / (K - 1), (K - 1) // 2
        self.t_min = -self.span
        self.slope_min, self.slope_max = float(slope_min), float(slope_max)
        if not (self.slope_min <= 0.0 <= self.slope_max):
            raise ValueError("slope_min <= 0 <= slope_max is required (the zero spline must be feasible)")
        if parametrization not in ("slopes", "coefficients"):
            raise ValueError(parametrization)
        if scales not in ("crr", "crr_sigma"):
            raise ValueError(scales)
        self.parametrization, self.scales, self.antisymmetric = parametrization, scales, bool(antisymmetric)
        self.conditioned = scales == "crr_sigma"  # needs the noise level
        if parametrization == "slopes":  # odd phi <=> mirror-symmetric slopes: only the right half is a parameter
            n = self.mid if self.antisymmetric else K - 1
        else:
            n = K
        self.shared = bool(shared)
        self.theta = nn.Parameter(torch.zeros(1 if self.shared else self.G, n))  # zero spline at init
        self.learn_scales = bool(learn_scales)
        self.log_mu = nn.Parameter(torch.tensor(math.log(float(mu_init))), requires_grad=self.learn_scales)
        self.mu_max = float(mu_max)
        if scales == "crr":
            self.log_alpha = nn.Parameter(math.log(float(alpha_init)) * torch.ones(1, self.G, 1, 1, 1), requires_grad=self.learn_scales)
        else:
            self.n_sigma = int(sigma_knots)
            self.sigma_min, self.sigma_max = float(sigma_range[0]), float(sigma_range[1])
            self.sigma_h = (self.sigma_max - self.sigma_min) / (self.n_sigma - 1)
            self.default_sigma = float(default_sigma)
            self.sigma_spline = nn.Parameter(float(sigma_spline_init) * torch.ones(self.G, self.n_sigma), requires_grad=self.learn_scales)
        self.compile = bool(compile)
        self.register_buffer("offsets", (torch.arange(self.G) * K).view(1, self.G, 1, 1, 1).to(torch.int32), persistent=False)
        self.register_buffer("mid_t", torch.tensor(float(self.mid)), persistent=False)

    # ---- scales ----
    def sigma_log_scale(self, sigma):
        """(B, G) values s_c(sigma_b) of the per-channel linear splines of the noise level (sigma in 0-255 units)."""
        t = (sigma.clamp(self.sigma_min, self.sigma_max) - self.sigma_min) / self.sigma_h
        i0 = t.floor().clamp(max=self.n_sigma - 2).long()
        frac = (t - i0).view(-1, 1).to(self.sigma_spline.dtype)
        c = self.sigma_spline
        return torch.lerp(c[:, i0].T, c[:, i0 + 1].T, frac)

    def _scales(self, sigma=None):
        """(a, o, mu): input scale a and output scale o, (1 or B, G, 1, 1, 1), and the global strength mu."""
        mu = torch.exp(self.log_mu).clamp(max=self.mu_max)
        if self.scales == "crr":
            a = torch.exp(self.log_alpha)
            return a, 1.0 / a, mu
        if sigma is None:
            sigma = torch.tensor([self.default_sigma], device=self.sigma_spline.device, dtype=self.sigma_spline.dtype)
        sigma = sigma.to(self.sigma_spline.dtype).flatten()
        # the official parametrisation exp(spline(sigma)) / (sigma + eps); contiguous so that broadcasts keep the input layout
        a = (torch.exp(self.sigma_log_scale(sigma)) / (sigma.view(-1, 1) + 1e-5)).reshape(-1, self.G, 1, 1, 1).contiguous()
        return a, 1.0 / a, mu

    def gain(self):
        """(G,) multiplier of the spline slopes in the effective Jacobian (= mu, sigma-free)."""
        a, o, mu = self._scales()
        return (mu * o * a)[0].flatten()

    def sigma_curves(self, n=61):
        """(sigmas (n,), alpha (G, n)): the input scales alpha_c(sigma) over the noise range (crr_sigma only)."""
        s = torch.linspace(max(self.sigma_min, 0.5), self.sigma_max, n, device=self.sigma_spline.device, dtype=self.sigma_spline.dtype)
        with torch.no_grad():
            a, _, _ = self._scales(s)
        return s, a.view(n, self.G).T

    # ---- the spline ----
    def slopes(self):
        """(G, K-1) slopes of phi_c on the cells, within [slope_min, slope_max]; mirror-symmetric when antisymmetric.
        With a shared spline the single row is expanded to G rows."""
        if self.parametrization == "slopes":
            s = self.theta.clamp(self.slope_min, self.slope_max)
            s = torch.cat([s.flip(1), s], dim=1) if self.antisymmetric else s
        else:
            th = 0.5 * (self.theta - self.theta.flip(1)) if self.antisymmetric else self.theta
            s = ((th[:, 1:] - th[:, :-1]) / self.h).clamp(self.slope_min, self.slope_max)
        return s.expand(self.G, -1) if self.shared else s

    def coefficients(self):
        """(G, K) knot values phi_c(t_i), t_i = -span + i h, anchored at phi_c(0) = 0."""
        sl = self.slopes()
        c = torch.cat([sl.new_zeros(self.G, 1), torch.cumsum(sl, dim=1) * self.h], dim=1)
        return c - c[:, self.mid:self.mid + 1]

    def forward(self, u, sigma=None):
        a, o, mu = self._scales(sigma)
        c = self.coefficients()
        if o.shape[0] == 1:  # per-channel output scale: folded into the table
            table = (c * (mu * o).view(self.G, 1)).reshape(-1)
            return spline_lookup(u, a / self.h, self.mid_t, table, self.offsets, self.K, self.compile)
        table = (c * mu).reshape(-1)  # per-sample scales (sigma per sample)
        return spline_lookup(u, a / self.h, self.mid_t, table, self.offsets, self.K, self.compile) * o

    def potential(self, u, sigma=None):
        """(B,) sum over channels and pixels of psi_c(u), the potential whose gradient is the activation."""
        a, o, mu = self._scales(sigma)
        c = self.coefficients()
        P = torch.cat([c.new_zeros(self.G, 1), torch.cumsum(0.5 * (c[:, 1:] + c[:, :-1]) * self.h, dim=1)], dim=1)
        P = P - P[:, self.mid:self.mid + 1]  # int_0^{t_i} phi_c (trapezoids are exact for a linear spline)
        t = torch.addcmul(self.mid_t, u, a / self.h).clamp(0, self.K - 1)
        idx = t.detach().int().clamp_(max=self.K - 2)
        r = (t - idx) * self.h  # offset from the left knot of the cell, scaled units
        base = idx.add_(self.offsets).view(-1)
        cf, Pf = c.reshape(-1), P.reshape(-1)
        cl = cf.index_select(0, base).view_as(t)
        cr = cf[1:].index_select(0, base).view_as(t)
        Pl = Pf.index_select(0, base).view_as(t)
        inside = Pl + cl * r + (0.5 / self.h) * (cr - cl) * r * r
        v = u * a  # linear continuation of the integral outside the knots (phi is constant there)
        ext = (v - self.span).clamp(min=0) * c[:, -1].view(1, self.G, 1, 1, 1) + \
              (v + self.span).clamp(max=0) * c[:, 0].view(1, self.G, 1, 1, 1)
        return (mu * o / a * (inside + ext)).sum(dim=(1, 2, 3, 4))

    # ---- bounds / certificate ----
    def max_block_norm(self):
        return self.gain() * self.slopes().abs().max(dim=1).values

    def block_eig_range(self):
        g = self.gain()
        sl = self.slopes()
        return g * sl.min(dim=1).values, g * sl.max(dim=1).values

    def penalties(self):
        """TV2 of the splines as in CRR: sum over channels and cells of |s_{i+1} - s_i|. Weight: train.penalties.tv2."""
        sl = self.slopes()
        return {"tv2": (sl[:, 1:] - sl[:, :-1]).abs().sum()}

    @torch.no_grad()
    def certificate(self, rho=0.0):
        lo, hi = self.block_eig_range()
        sl = self.slopes()
        smin, smax = sl.min(dim=1).values, sl.max(dim=1).values
        a, _, mu = self._scales()
        return {"monotone": bool(lo.min() >= -rho), "conservative": True,
                "min_slope": float(smin.min()), "max_slope": float(smax.max()),
                "frac_nonmonotone": float((smin < 0).float().mean()), "mu": float(mu), "alpha_mean": float(a.mean())}

    def evaluate_on_grid(self, n=401, margin=1.25, sigma=None):
        """Effective nonlinearities on physical inputs: u (G, n) spanning +-margin*span/alpha_c, and Phi_c(u) (G, n)
        (at `sigma`, 0-255 units, or the default noise level for conditioned scales)."""
        sig = None if sigma is None else torch.tensor([float(sigma)], device=self.theta.device)
        a, _, _ = self._scales(sig)
        a = a.flatten()
        lin = torch.linspace(-margin, margin, n, device=a.device, dtype=a.dtype)
        u = self.span * lin.view(1, n) / a.view(self.G, 1)
        with torch.no_grad():
            out = self.forward(u.view(1, self.G, 1, 1, n), sigma=sig)
        return u, out.view(self.G, n)

    def extra_repr(self):
        s = (f"G={self.G}x{self.k}, {self.K} knots on [{self.t_min:g}, {self.span:g}], slopes in [{self.slope_min:g}, {self.slope_max:g}], "
             f"scales={self.scales}{'' if self.learn_scales else ' (fixed)'}, param={self.parametrization}, "
             f"antisymmetric={self.antisymmetric}, shared={self.shared}, compile={self.compile}")
        if self.conditioned:
            s += f", sigma spline: {self.n_sigma} knots on [{self.sigma_min:g}, {self.sigma_max:g}], default sigma {self.default_sigma:g}"
        return s


def build_activation(cfg, groups, group_size):
    import inspect
    cfg = dict(cfg or {})
    typ = cfg.pop("type", "spline")
    cls = {"spline": LinearSplineActivation}.get(typ)
    if cls is None:
        raise ValueError(f"unknown activation type {typ}: this repository ships the linear spline only")
    allowed = set(inspect.signature(cls.__init__).parameters) - {"self", "groups", "group_size"}
    unknown = set(cfg) - allowed
    if unknown:
        print(f"[build_activation] ignoring keys {sorted(unknown)} not accepted by {cls.__name__}")
    return cls(groups, group_size, **{k: v for k, v in cfg.items() if k in allowed})
