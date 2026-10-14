#!/usr/bin/env python
"""Figure: skew fraction of the Jacobian along the path from a natural image (t = 0) to white noise of the same scale (t = 1)."""
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results/jacobian_symmetry_sweep.json")
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "figures/jacobian_sweep.pdf")
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "axes.linewidth": 0.5, "lines.linewidth": 1.0, "pdf.fonttype": 42})
R = json.load(open(src)); t = np.array(R["t"]); C = np.array(R["curves"])
fig, ax = plt.subplots(figsize=(3.4, 1.4))
ax.fill_between(t, C.min(0), C.max(0), color="#1f77b4", alpha=0.2, lw=0, label="range over images")
ax.plot(t, C.mean(0), color="#1f77b4", marker="o", ms=2.5, label="mean")
ax.set_xlabel(r"$t$  ($0$: natural image, $1$: white noise of the same norm)", labelpad=1)
ax.set_ylabel(r"skew fraction $s$", labelpad=1); ax.set_xlim(0, 1); ax.set_ylim(0, max(0.16, 1.08 * float(C.max())))
ax.legend(frameon=False, handlelength=1.4, loc="lower right")
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
fig.tight_layout(pad=0.3); fig.savefig(out); print("saved", out, "t=0 mean", C.mean(0)[0], "t=1 mean", C.mean(0)[-1])
