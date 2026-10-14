"""Linear analysis operator W (multi-layer convolution collapsed to one filter bank).

W x = H (K_3 K_2 K_1 x) / s   with K_i zero-padded convolutions (first one zero-mean), H an optional 1x1 channel-mixing
layer and s the spectral norm of the whole operator (circular estimate through the FFT of the impulse response, as in the
CRR/CSR code), so that ||W||_2 = 1. The transpose is exact for the zero-padded convolutions.

For the untied model the synthesis operator is built from the same weights with the LAST linear layer perturbed:
V_ = (K_3 + Delta) K_2 K_1 (or (H + Delta) K_3 K_2 K_1 when a 1x1 layer is present), see `with_last_delta*`.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as P


def weighted_sq_norm_from_impulse(imp, weights=None, fft_size=256):
    """max_f sum_c weights_c |imp_c(f)|^2 for an impulse tensor (C, k, k)."""
    p = torch.fft.fft2(imp, s=(fft_size, fft_size)).abs().pow(2)
    if weights is not None:
        p = p * weights.view(-1, 1, 1)
    return p.sum(dim=0).max()


class ZeroMean(nn.Module):
    """Zero-mean kernels for each (out, in) channel pair."""

    def forward(self, w):
        return w - w.mean(dim=(2, 3), keepdim=True)


class MultiConv(nn.Module):
    def __init__(self, channels=(1, 4, 8, 60), kernel_sizes=(5, 5, 5), hidden_1x1=False, hidden_out=None,
                 zero_mean=True, fft_size=256, normalize_full=True):
        super().__init__()
        channels, kernel_sizes = list(channels), list(kernel_sizes)
        assert len(kernel_sizes) == len(channels) - 1 and channels[0] == 1, "single-channel input expected"
        self.convs = nn.ModuleList(
            nn.Conv2d(channels[i], channels[i + 1], kernel_sizes[i], padding=kernel_sizes[i] // 2, bias=False)
            for i in range(len(kernel_sizes))
        )
        if zero_mean:
            P.register_parametrization(self.convs[0], "weight", ZeroMean())
        self.hidden = None
        self.out_channels = channels[-1]
        if hidden_1x1:
            hidden_out = hidden_out or channels[-1]
            self.hidden = nn.Conv2d(channels[-1], hidden_out, 1, bias=False)
            self.out_channels = hidden_out
        self.fft_size = int(fft_size)
        self.kernel_size = sum(kernel_sizes) - len(kernel_sizes) + 1  # effective support
        dirac = torch.zeros(1, 1, self.kernel_size, self.kernel_size)
        dirac[0, 0, self.kernel_size // 2, self.kernel_size // 2] = 1.0
        self.register_buffer("dirac", dirac, persistent=False)
        self.normalize_full = bool(normalize_full)  # divide by the spectral norm of the full W (convs + 1x1)
        self._sn_cache = None  # set by RidgeField.cached() for the duration of a fixed-point solve

    # ---- spectral normalisation ----
    def _stack(self, x):
        for c in self.convs:
            x = c(x)
        return x

    def _stack_t(self, u):
        for c in reversed(self.convs):
            u = F.conv_transpose2d(u, c.weight, padding=c.padding)
        return u

    def stack_spectral_norm(self):
        """Spectral norm (circular FFT estimate) of the un-normalised operator: the convolution stack alone, or the
        convolution stack followed by the 1x1 layer when normalize_full=True. W is divided by this number, so that
        ||W||_2 = 1 (full) or ||convs||_2 = 1 (stack only, legacy behaviour)."""
        if self._sn_cache is not None:
            return self._sn_cache
        imp = self._stack(self.dirac)  # (1, C, k, k)
        if self.normalize_full and self.hidden is not None:
            imp = self.hidden(imp)
        fr = torch.fft.fft2(imp, s=(self.fft_size, self.fft_size))
        return fr.abs().pow(2).sum(dim=1).max().sqrt()

    def set_cache(self, on=True):
        """Freeze the (differentiable) spectral-norm estimate for the current parameters."""
        self._sn_cache = self.stack_spectral_norm() if on else None

    # ---- forward / transpose ----
    def forward(self, x):
        u = self._stack(x / self.stack_spectral_norm())
        if self.hidden is not None:
            u = self.hidden(u)
        return u

    def transpose(self, u):
        if self.hidden is not None:
            u = F.conv_transpose2d(u, self.hidden.weight)
        return self._stack_t(u) / self.stack_spectral_norm()

    # ---- effective filters and their spectrum ----
    def impulse(self):
        """Effective filters, shape (C_out, k, k): response of W to a centred Dirac."""
        return self.forward(self.dirac)[0]

    def freq_response(self):
        """(C_out, S, S) complex DFT of the effective filters (circular approximation)."""
        return torch.fft.fft2(self.impulse(), s=(self.fft_size, self.fft_size))

    def weighted_sq_norm(self, weights=None):
        """max_f sum_c weights_c |w_c(f)|^2  = ||W^T diag(weights) W||_2 (circular approximation).
        weights: (C_out,) nonnegative tensor or None (all ones -> ||W||_2^2)."""
        p = self.freq_response().abs().pow(2)  # (C, S, S)
        if weights is not None:
            p = p * weights.view(-1, 1, 1)
        return p.sum(dim=0).max()

    # ---- un-normalised operator with the last linear layer perturbed: V_ = (K_L + Delta) K_{L-1} ... K_1 ----
    def last_layer(self):
        """The last linear layer (the 1x1 layer when present, else the last convolution); Delta has its weight's shape."""
        return self.hidden if self.hidden is not None else self.convs[-1]

    def with_last_delta(self, x, delta):
        if self.hidden is not None:
            return F.conv2d(self._stack(x), self.hidden.weight + delta)
        convs = list(self.convs)
        for c in convs[:-1]:
            x = c(x)
        return F.conv2d(x, convs[-1].weight + delta, padding=convs[-1].padding)

    def with_last_delta_t(self, u, delta):
        if self.hidden is not None:
            return self._stack_t(F.conv_transpose2d(u, self.hidden.weight + delta))
        convs = list(self.convs)
        u = F.conv_transpose2d(u, convs[-1].weight + delta, padding=convs[-1].padding)
        for c in reversed(convs[:-1]):
            u = F.conv_transpose2d(u, c.weight, padding=c.padding)
        return u

    def with_last_delta_impulse(self, delta):
        return self.with_last_delta(self.dirac, delta)[0]

    # ---- collapsed form: one k x k convolution per channel (exact in the interior, differs within ~k/2 pixels of
    #      the border from the per-layer zero-padded composition) ----
    @staticmethod
    def kernel_from_impulse(imp):
        """conv2d weight (C, 1, k, k) whose cross-correlation reproduces the impulse response imp (C, k, k)."""
        return torch.flip(imp, dims=(1, 2))[:, None]

    @staticmethod
    def apply_kernel(x, kernel):
        return F.conv2d(x, kernel, padding=kernel.shape[-1] // 2)

    @staticmethod
    def apply_kernel_t(u, kernel):
        return F.conv_transpose2d(u, kernel, padding=kernel.shape[-1] // 2)
