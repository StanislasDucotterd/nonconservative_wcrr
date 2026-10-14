"""Figures of learned activations (files and TensorBoard)."""
import math
import torch


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _fmt(ax, nx=5, ny=5):
    """Plain tick labels (no 1e-2 offsets), few ticks."""
    from matplotlib.ticker import MaxNLocator, FuncFormatter
    for axis, n in ((ax.xaxis, nx), (ax.yaxis, ny)):
        axis.set_major_locator(MaxNLocator(n))
        axis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:.3g}"))


def spline_stats(act):
    with torch.no_grad():
        a, _, lam = act._scales()
        return {"alpha": a.flatten().cpu(), "gain": act.gain().cpu(), "lambda": float(lam), "slopes": act.slopes().cpu()}


@torch.no_grad()
def input_quantiles(field, x, qs=(0.5, 0.9)):
    """(len(qs), G) quantiles of |(Wx)_c| over the pixels of the image(s) x: where the data lives for each channel."""
    u = field.W(x)
    B = u.shape[0]
    u = u.abs().view(B, field.G, -1).transpose(0, 1).reshape(field.G, -1)
    return u.quantile(torch.tensor(list(qs), device=u.device, dtype=u.dtype), dim=1).cpu()


def shared_spline_figure(field):
    """The single spline phi(v) of a shared-spline activation on its own coordinate v in [-span, span]
    (the per-channel scales are in the sigma-scaling / stats figures)."""
    plt = _mpl()
    act = field.act
    with torch.no_grad():
        c = act.coefficients()[0].cpu()
        sl = act.slopes()[0].cpu()
        lam = float(act._scales()[2])
    t = torch.linspace(act.t_min, act.span, act.K)
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.plot(t, c, lw=1.4)
    ax.axhline(0, color="gray", lw=0.4)
    ax.axvline(0, color="gray", lw=0.4)
    ax.set_xlim(act.t_min, act.span)
    ax.set_xlabel("v = alpha_c(sigma) u")
    ax.set_ylabel("phi(v)")
    ax.set_title(f"shared spline: slopes in [{float(sl.min()):+.2f}, {float(sl.max()):+.2f}] (allowed [{act.slope_min:g}, {act.slope_max:g}]); lambda = {lam:.4g}", fontsize=8)
    _fmt(ax, 7, 7)
    fig.tight_layout()
    return fig


def spline_figure(field, ncols=8, n=801, data_quantiles=None, xlim=0.1):
    """One panel per channel: Phi_c(u) = lam * phi_c(a_c u) / a_c on physical inputs u in the fixed window [-xlim, xlim]
    (same for all channels; the function is constant beyond the knots +-span/a_c), y-axis zoomed to each channel's own
    range. With data_quantiles (from input_quantiles) the band |u| <= 90% quantile of |Wx| is shaded.
    For a shared spline (one phi for all channels) the single spline is drawn instead."""
    plt = _mpl()
    act = field.act
    if getattr(act, "shared", False):
        return shared_spline_figure(field)
    st = spline_stats(act)
    a, lam, sl = st["alpha"], st["lambda"], st["slopes"]
    G = a.shape[0]
    q90 = None if data_quantiles is None else data_quantiles[-1].cpu()
    with torch.no_grad():
        u = torch.linspace(-xlim, xlim, n)
        dev = act.theta.device
        phi = act(u.to(dev).view(1, 1, 1, 1, n).expand(1, G, 1, 1, n).contiguous()).view(G, n).cpu()
    nrows = math.ceil(G / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.3 * ncols, 2.0 * nrows), squeeze=False)
    for c in range(nrows * ncols):
        ax = axes[c // ncols, c % ncols]
        if c >= G:
            ax.axis("off")
            continue
        if q90 is not None:
            ax.axvspan(-float(q90[c]), float(q90[c]), color="0.92", zorder=0)
        ax.plot(u, phi[c], lw=1.1)
        ax.axhline(0, color="gray", lw=0.4)
        ax.axvline(0, color="gray", lw=0.4)
        ymax = float(phi[c].abs().max()) * 1.1 + 1e-12
        ax.set_ylim(-ymax, ymax)
        ax.set_xlim(float(u[0]), xlim)
        ax.set_title(f"c{c}  alpha {a[c]:.2f}  slopes [{sl[c].min():+.2f}, {sl[c].max():+.2f}]", fontsize=7, pad=2)
        ax.tick_params(labelsize=5)
        _fmt(ax)
        if c // ncols == nrows - 1:
            ax.set_xlabel("u (input units)", fontsize=6)
    extra = f";  lambda = {lam:.4g}" if act.scales == "crr" else ""
    band = ";  grey: |u| <= 90% quantile of |Wx|" if q90 is not None else ""
    fig.suptitle(f"Phi_c(u) for u in [-{xlim:g}, {xlim:g}] (constant beyond the knots +-{act.span:g}/alpha_c), y zoomed per channel{band}{extra}", fontsize=9)
    fig.tight_layout()
    return fig


def spline_overlay_figure(field):
    """All raw splines phi_c(v) on their own coordinate v in [-span, span] (shape only: alpha and lambda factored out)."""
    plt = _mpl()
    act = field.act
    with torch.no_grad():
        c = act.coefficients().cpu()
        sl = act.slopes()
    t = torch.linspace(act.t_min, act.span, act.K)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for g in range(c.shape[0]):
        ax.plot(t, c[g], lw=0.8, alpha=0.7)
    ax.axhline(0, color="gray", lw=0.4)
    ax.axvline(0, color="gray", lw=0.4)
    ymax = float(c.abs().max()) * 1.1 + 1e-12
    ax.set_ylim(-ymax, ymax)
    ax.set_xlim(act.t_min, act.span)
    ax.set_xlabel("v = alpha_c u")
    ax.set_ylabel("phi_c(v)")
    ax.set_title(f"{c.shape[0]} raw splines on [-{act.span:g}, {act.span:g}]; slopes allowed in [{act.slope_min:g}, {act.slope_max:g}], "
                 f"learned in [{float(sl.min()):+.2f}, {float(sl.max()):+.2f}]", fontsize=8)
    _fmt(ax, 7, 7)
    fig.tight_layout()
    return fig


def sigma_scaling_figure(field):
    """alpha_g(sigma) = exp(s_g(sigma)) / sigma of a sigma-conditioned spline activation over the noise range (log scale),
    with the knot range of the splines in input units, span / alpha_g(sigma), on the right axis."""
    plt = _mpl()
    act = field.act
    s, a = act.sigma_curves()
    s, a = s.cpu(), a.cpu()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for g in range(a.shape[0]):
        axes[0].semilogy(s, a[g], lw=0.8, alpha=0.7)
        axes[1].semilogy(s, act.span / a[g], lw=0.8, alpha=0.7)
    axes[0].set_xlabel("sigma (0-255 units)"); axes[0].set_ylabel("alpha_g(sigma)"); axes[0].set_title("input scales alpha_g(sigma)", fontsize=9)
    axes[1].set_xlabel("sigma (0-255 units)"); axes[1].set_ylabel("span / alpha_g"); axes[1].set_title("knot range in input units", fontsize=9)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_linear_splines(field, path, ncols=10, data_quantiles=None):
    """Save the per-channel figure to `path`; returns per-channel stats."""
    plt = _mpl()
    fig = spline_figure(field, ncols=ncols, data_quantiles=data_quantiles)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    st = spline_stats(field.act)
    sl = st["slopes"]
    return {"min_slope": sl.min(dim=1).values.tolist(), "max_slope": sl.max(dim=1).values.tolist(),
            "alpha": st["alpha"].tolist(), "gain": st["gain"].tolist(), "lambda": st["lambda"],
            "range_u": (field.act.span / st["alpha"]).tolist()}
