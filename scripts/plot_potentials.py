#!/usr/bin/env python3
"""Fig. 1: V(r) for the five discrete-color pair classes."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from diquark_md.potentials import pair_potential

ALPHA, LAM_D, R0 = 0.4, 0.4, 0.2
BIG_CUT = 1e9  # unshifted for display

# (label, C_ij, color)  -- categorical palette, fixed order by entity
CLASSES = [
    (r"$q\bar q$ matched ($r\bar r$): $\frac{1}{3}\mathbf{1}\oplus\frac{2}{3}\mathbf{8}$", -2/3, "#2a78d6"),
    (r"$qq$ different ($rg$): $\frac{1}{2}\bar{\mathbf{3}}\oplus\frac{1}{2}\mathbf{6}$", -1/3, "#eb6834"),
    (r"$q\bar q$ mismatched ($r\bar g$): $\mathbf{8}$", +1/3, "#1baf7a"),
    (r"$qq$ same ($rr$): $\mathbf{6}$", +2/3, "#eda100"),
]

r = np.linspace(0.0, 2.0, 400)
fig, ax = plt.subplots(figsize=(4.2, 3.2), constrained_layout=True)
for label, c_ij, col in CLASSES:
    v = np.array([pair_potential(ri, c_ij, ALPHA, LAM_D, R0, BIG_CUT) * 1e3
                  for ri in r])
    ax.plot(r, v, color=col, lw=1.8, label=label)
ax.axhline(0, color="#8a8a8a", lw=0.8)
ax.set_xlabel(r"$r$ [fm]")
ax.set_ylabel(r"$V(r)$ [MeV]")
ax.set_ylim(-60, 35)
ax.legend(fontsize=7.2, frameon=False, loc="lower right")
ax.spines[["top", "right"]].set_visible(False)
fig.savefig("paper/figs/fig1_potentials.pdf")
fig.savefig("paper/figs/fig1_potentials.png", dpi=200)
print("saved fig1")
