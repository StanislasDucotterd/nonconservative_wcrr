#!/usr/bin/env python
"""Figures of the paper (PDF, figures/): the 64 filter pairs (W_c over V_c, grey scale) and the learned spline with
its integral, for one checkpoint.
    python scripts/paper_figures.py [trained_models/<run>/last.pth]"""
import json
import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ncr.trainer import load_checkpoint
OUT = os.path.join(ROOT, "figures")
CK = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "trained_models/nonconservative_wcrr/last.pth")
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "axes.linewidth": 0.5, "lines.linewidth": 1.0, "pdf.fonttype": 42})
os.makedirs(OUT, exist_ok=True)
field, model, cfg = load_checkpoint(CK, "cpu")
with torch.no_grad():
    W = field.W.impulse(); V = field.V_impulse()
    cos = (W * V).flatten(1).sum(1) / (W.flatten(1).norm(dim=1) * V.flatten(1).norm(dim=1))
order = torch.argsort(cos, descending=True).tolist()
Wd, Vd = W.numpy(), V.numpy()
# ---- figure 1: all 64 pairs, 16 columns x 4 rows of (W over V), full page width (7.0 in)
ncol, nrow = 16, 4
FW = 7.0; pitch = FW / ncol; side = 0.72 * pitch; gap = 0.02; bar = 0.03; rowh = 2 * side + gap + bar + 0.07
FH = nrow * rowh
fig = plt.figure(figsize=(FW, FH))
for n, c in enumerate(order):
    r, k = divmod(n, ncol)
    x0 = k * pitch + (pitch - side) / 2
    yW = FH - r * rowh - side; yV = yW - gap - side
    m = max(abs(Wd[c]).max(), abs(Vd[c]).max())
    for y, img in ((yW, Wd[c]), (yV, Vd[c])):
        ax = fig.add_axes([x0 / FW, y / FH, side / FW, side / FH]); ax.imshow(img, cmap="gray", vmin=-m, vmax=m, interpolation="nearest"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_linewidth(0.3)
    col = "#1f77b4" if cos[c] > 0.9 else ("#d62728" if cos[c] < -0.5 else "#7f7f7f")
    b = fig.add_axes([x0 / FW, (yV - gap - bar) / FH, side / FW, bar / FH]); b.set_facecolor(col); b.set_xticks([]); b.set_yticks([])
    for sp in b.spines.values(): sp.set_visible(False)
fig.savefig(os.path.join(OUT, "filter_pairs.pdf")); plt.close(fig)
print("pairs:", int((cos > 0.9).sum()), "aligned,", int((cos < -0.5).sum()), "flipped,", int(((cos >= -0.5) & (cos <= 0.9)).sum()), "other; cos range of flipped", float(cos[cos < -0.5].max()), float(cos[cos < -0.5].min()))
# ---- figure 2: the spline and its integral
act = field.act
with torch.no_grad():
    coef = act.coefficients()[0].numpy(); t = (np.arange(act.K) - act.mid) * float(act.h)
v = np.linspace(t[0], t[-1], 2001); phi = np.interp(v, t, coef)
psi = np.concatenate([[0.0], np.cumsum(0.5 * (phi[1:] + phi[:-1]) * np.diff(v))]); psi -= np.interp(0.0, v, psi)
fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.45))
for ax, y, lab in ((axes[0], phi, r"$\varphi(v)$"), (axes[1], psi, r"$\psi(v)$")):
    ax.plot(v, y, color="#1f77b4"); ax.axhline(0, color="0.7", lw=0.4); ax.axvline(0, color="0.7", lw=0.4)
    ax.set_xlabel(r"$v$", labelpad=1); ax.set_ylabel(lab, labelpad=1); ax.set_xticks([-0.1, 0, 0.1])
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2)); ax.yaxis.get_offset_text().set_fontsize(6)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
fig.tight_layout(pad=0.3); fig.savefig(os.path.join(OUT, "spline.pdf")); plt.close(fig)
print("figures written to", OUT)
