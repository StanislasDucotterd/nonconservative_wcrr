"""The field F whose fixed-point equation x + F(x) = y defines the denoiser.

    F(x) = V^T Phi(W x)

W: MultiConv analysis operator with ||W||_2 = 1; Phi: the learnable linear splines of CRR / WCRR (ncr/activations.py);
V: synthesis operator, either V = W (tied, conservative field) or, with untied='delta_last',
    V = (K_L + Delta) K_{L-1} ... K_1 / ||.||_2     (W = K_L ... K_1 / ||.||_2, same weights, Delta free, zero at init)
so that V = W exactly at initialisation and the model stays 1-Lipschitz on both sides.

Guarantee classes (reported by `guarantee()`):
    conservative                  F = grad R, R convex            -> prox denoiser (firmly nonexpansive)
    monotone (certified)          Sym J_F >= 0                    -> resolvent of a maximally monotone operator (FNE)
    rho-weakly monotone           Sym J_F >= -rho I, rho < 1      -> unique fixed point, damped iteration converges
    none                          only diagnostics
"""
import contextlib
import torch
import torch.nn as nn
from .linear import MultiConv, weighted_sq_norm_from_impulse
from .activations import build_activation


class RidgeField(nn.Module):
    def __init__(self, channels=(1, 4, 8, 60), kernel_sizes=(5, 5, 5), hidden_1x1=False, hidden_out=None,
                 zero_mean=True, group_size=1, activation=None, untied=False, fft_size=256, normalize_full=True):
        super().__init__()
        self.W = MultiConv(channels, kernel_sizes, hidden_1x1=hidden_1x1, hidden_out=hidden_out,
                           zero_mean=zero_mean, fft_size=fft_size, normalize_full=normalize_full)
        C = self.W.out_channels
        assert C % group_size == 0, "number of channels must be a multiple of the group size"
        self.k = int(group_size)
        self.G = C // self.k
        if untied in (False, None, "tied"):
            self.untied_mode = "tied"
        elif untied == "delta_last":
            self.untied_mode = "delta_last"
        else:
            raise ValueError(f"untied={untied!r}: supported values are false (V = W) and 'delta_last'")
        # Delta: perturbation of the last linear layer's weight in the synthesis operator, zero at init (V = W)
        self.delta = nn.Parameter(torch.zeros_like(self.W.last_layer().weight)) if self.untied else None
        self.act = build_activation(activation, self.G, self.k)
        self._sigma = None  # noise level(s) for sigma-conditioned activations, set by the denoiser (see set_sigma)
        self.use_collapsed = False  # evaluation mode: apply W and V as single effective-kernel convolutions
        self._cache = None  # dict with 'lip' (and 'sV', collapsed kernels) while inside cached()

    @contextlib.contextmanager
    def cached(self):
        """Within this context the spectral norms and Lipschitz bound are computed once (with graph) and reused
        by every application of F. Used by the denoiser for the whole fixed-point solve + differentiable step."""
        try:
            self.W.set_cache(True)
            self._cache = {}
            if self.untied:
                self._cache["sV"] = self._sV()
            self._cache["lip"] = self._lip_bound()
            if self.use_collapsed:
                self._cache["kW"] = MultiConv.kernel_from_impulse(self.W.impulse())
                self._cache["kV"] = MultiConv.kernel_from_impulse(self.V_impulse()) if self.untied else self._cache["kW"]
            yield
        finally:
            self.W.set_cache(False)
            self._cache = None

    # ---- noise level (sigma-conditioned activations only) ------------------------------------------
    @property
    def conditioned(self):
        return bool(getattr(self.act, "conditioned", False))

    def set_sigma(self, sigma):
        """Noise level(s) in 0-255 units for a sigma-conditioned activation: None (the activation's default), a float,
        or a (B,) tensor with one value per sample of the next batch. Ignored by unconditioned activations."""
        if sigma is None:
            self._sigma = None
        elif torch.is_tensor(sigma):
            self._sigma = sigma.detach().flatten().to(self.W.dirac.device)
        else:
            self._sigma = torch.tensor([float(sigma)], device=self.W.dirac.device)

    def _act_kwargs(self, x, idx):
        if not self.conditioned or self._sigma is None:
            return {}
        s = self._sigma
        if s.numel() == 1:
            return {"sigma": s}
        if idx is not None:
            return {"sigma": s[idx]}
        if s.numel() == x.shape[0]:
            return {"sigma": s}
        raise RuntimeError(f"set_sigma holds {s.numel()} noise levels but the batch has {x.shape[0]} samples and no indices were given")

    # ---- structure -------------------------------------------------------------------------------
    @property
    def untied(self):
        return self.untied_mode != "tied"

    @property
    def conservative(self):
        return (not self.untied) and self.act.is_conservative

    def _sV(self):
        """Spectral norm (circular FFT estimate) of the un-normalised synthesis operator V_."""
        if self._cache is not None and "sV" in self._cache:
            return self._cache["sV"]
        return torch.sqrt(weighted_sq_norm_from_impulse(self.W.with_last_delta_impulse(self.delta), None, self.W.fft_size))

    def synthesis_transpose(self, phi):
        """V^T phi."""
        if self.untied:
            return self.W.with_last_delta_t(phi, self.delta) / self._sV()
        return self.W.transpose(phi)

    def V_impulse(self):
        """Effective synthesis filters (C, k, k)."""
        if self.untied:
            return self.W.with_last_delta_impulse(self.delta) / self._sV()
        return self.W.impulse()

    def guarantee(self):
        cert = self.act.certificate()
        tied_kind = "convex" if self.act.is_conservative else "monotone"  # tied + monotone activation
        if (not self.untied) and cert.get("monotone", None) is True:
            return tied_kind
        wb = self.weak_monotone_bound()
        if wb is None:
            return "none (diagnostics only)"
        wb = float(wb)
        if wb == 0.0:
            return "convex (V == W)" if self.untied else tied_kind
        if self.untied:
            return f"{wb:.3g}-weakly-monotone (certified by polarisation bound)" if wb < 1 else f"none (certified modulus {wb:.3g} >= 1)"
        weak = "weakly-convex" if self.act.is_conservative else "weakly-monotone"
        return f"{wb:.3g}-{weak} (certified from the most negative slopes)" if wb < 1 else f"none (certified modulus {wb:.3g} >= 1)"

    # ---- operator --------------------------------------------------------------------------------
    def analysis(self, x):
        B, _, H, Wd = x.shape
        return self.W(x).view(B, self.G, self.k, H, Wd)

    def _collapsed_kernels(self):
        if self._cache is not None and "kW" in self._cache:
            return self._cache["kW"], self._cache["kV"]
        kW = MultiConv.kernel_from_impulse(self.W.impulse())
        kV = MultiConv.kernel_from_impulse(self.V_impulse()) if self.untied else kW
        return kW, kV

    def forward(self, x, idx=None):
        """F(x). idx: indices of the samples of x within the batch given to set_sigma (solvers evaluate subsets)."""
        B, _, H, Wd = x.shape
        kw = self._act_kwargs(x, idx)
        if self.use_collapsed:
            kW, kV = self._collapsed_kernels()
            u = MultiConv.apply_kernel(x, kW)
            phi = self.act(u.view(B, self.G, self.k, H, Wd), **kw).view(B, -1, H, Wd)
            return MultiConv.apply_kernel_t(phi, kV)
        u = self.W(x)
        phi = self.act(u.view(B, self.G, self.k, H, Wd), **kw).view(B, -1, H, Wd)
        return self.synthesis_transpose(phi)

    def potential(self, x, idx=None):
        """R(x) with F = grad R, only for conservative fields (None otherwise)."""
        if not self.conservative:
            return None
        u = self.W(x)
        B, _, H, Wd = x.shape
        return self.act.potential(u.view(B, self.G, self.k, H, Wd), **self._act_kwargs(x, idx))

    # ---- bounds ----------------------------------------------------------------------------------
    def lip_bound(self):
        """Upper bound on Lip(F): sqrt(||W^T D W|| ||V^T D V||), D = blockdiag(sigma_g I_k),
        sigma_g >= sup ||J_{Phi_g}||_2 (Cauchy-Schwarz over groups; circular FFT estimate of the norms).
        Tied: ||W^T D W|| (CRR, Prop. IV.1)."""
        if self._cache is not None:
            return self._cache["lip"]
        return self._lip_bound()

    def _lip_bound(self):
        sig = self.act.max_block_norm().repeat_interleave(self.k)  # (C,)
        LW = self.W.weighted_sq_norm(sig)
        if self.untied:
            LV = weighted_sq_norm_from_impulse(self.V_impulse(), sig, self.W.fft_size)
            return torch.sqrt(LW * LV)
        return LW

    def _block_eig_weights(self, detach=False):
        """Per-channel lo^- = max(-lo, 0) and hi^+ = max(hi, 0) from the activation's block eigenvalue range."""
        lo, hi = self.act.block_eig_range()
        lo, hi = lo.repeat_interleave(self.k), hi.repeat_interleave(self.k)
        if detach:
            lo, hi = lo.detach(), hi.detach()
        return torch.relu(-lo), torch.relu(hi)

    def weak_monotone_bound(self, detach_scales=False):
        """Certified hypomonotonicity modulus rho (F + rho I monotone) for a conservative group activation whose
        block Jacobians D_c are symmetric with eigenvalues in [lo_c, hi_c].  Split D = D^+ - D^- and polarise:
            Sym(J_F) = [(V+W)^T D (V+W) - (V-W)^T D (V-W)] / 4
                    >= -[(V+W)^T D^- (V+W) + (V-W)^T D^+ (V-W)] / 4
                    >= -1/4 max_f sum_c [ lo^-_c |v_c(f) + w_c(f)|^2 + hi^+_c |v_c(f) - w_c(f)|^2 ]  I   (FFT estimate).
        Tied (V = W): rho = ||W^T diag(lo^-) W|| = 0 for monotone activations (then the field is a convex gradient).
        None when the activation Jacobian is not symmetric (no certificate). Logged as `hypo_bound`."""
        if not self.act.is_conservative:
            return None
        neg, pos = self._block_eig_weights(detach_scales)
        has_neg = bool((neg > 0).any())
        if not self.untied:
            if not has_neg:
                return torch.zeros((), device=self.W.dirac.device)
            return self.W.weighted_sq_norm(neg)
        fs = self.W.fft_size
        Wi, Vi = self.W.impulse(), self.V_impulse()  # (C, k, k)
        S = torch.fft.fft2(Vi - Wi, s=(fs, fs)).abs().pow(2) * pos.view(-1, 1, 1)
        if has_neg:
            S = S + torch.fft.fft2(Vi + Wi, s=(fs, fs)).abs().pow(2) * neg.view(-1, 1, 1)
        return S.sum(dim=0).max() / 4

    def penalties(self):
        return dict(self.act.penalties())

    def param_groups(self):
        """Parameter groups for per-group learning rates (train.lr_mult): 'linear' (W and Delta) plus the activation's
        own groups ('activation' by default; e.g. MFoE splits off 'scales')."""
        linear = list(self.W.parameters()) + ([self.delta] if self.delta is not None else [])
        groups = {"linear": linear}
        if hasattr(self.act, "param_groups"):
            groups.update(self.act.param_groups())
        else:
            groups["activation"] = list(self.act.parameters())
        return groups

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self):
        return (f"RidgeField: C={self.W.out_channels} groups={self.G}x{self.k} untied={self.untied_mode} "
                f"act={type(self.act).__name__} "
                f"params={self.num_params()} guarantee={self.guarantee()}")


def build_field(cfg):
    return RidgeField(**dict(cfg))
