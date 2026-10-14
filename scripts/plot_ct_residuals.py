#!/usr/bin/env python
"""Figure: relative change of the SAGD iterates on one CT slice (from results/ct/changes_60.json).
    python scripts/plot_ct_residuals.py [results/ct/changes_60.json figures/convergence_ct.pdf [1500]]"""
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results/ct/changes_60.json")
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "figures/convergence_ct.pdf")
nmax = int(sys.argv[3]) if len(sys.argv) > 3 else None  # plot only the first nmax iterations
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "axes.linewidth": 0.5, "lines.linewidth": 1.0, "pdf.fonttype": 42})
R = json.load(open(src))
labels = {"ours": "ours", "wcrr": "WCRR"}
colors = {"ours": "#d62728", "wcrr": "#1f77b4"}
fig, ax = plt.subplots(figsize=(3.4, 1.6))
for m, r in R["models"].items():
    h = np.array(r["history"])[:nmax]; ax.semilogy(np.arange(1, len(h) + 1), h, color=colors.get(m, "k"), label=f"{labels.get(m, m)}: $\\lambda={r['lamb']:.3g}$, $\\sigma={r['sigma']:.3g}$")
ax.set_xlabel("iterations", labelpad=1)
ax.set_ylabel(r"$\|\mathbf{x}_{k+1}-\mathbf{x}_k\|/\|\mathbf{x}_k\|$" if R.get("stop") == "change" else r"$\|G(\mathbf{x}_k)\|/\|\mathbf{H}^\intercal\mathbf{y}\|$", labelpad=1)
if R.get("stop") == "change":
    ax.axhline(1e-5, color="0.5", lw=0.6, ls=(0, (4, 3)))
ax.legend(frameon=False, handlelength=1.2)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
fig.tight_layout(pad=0.3); fig.savefig(out); print("saved", out, {m: (len(r["history"]), r["history"][-1], r["psnr_final"]) for m, r in R["models"].items()})
