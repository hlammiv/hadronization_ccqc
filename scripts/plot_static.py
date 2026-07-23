#!/usr/bin/env python3
"""Fig. 2: static diquark-formation probability vs density.

Analytic hard-wall curves (Lebed and discrete-color counting) with MC
points and Lebed's published Table 2 values.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from diquark_md.observables import lebed_analytic, lebed_static_mc

BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#333333", "#8a8a8a"

n0_grid = np.logspace(-3, 2.2, 60)
ana_lebed = [lebed_analytic(n) for n in n0_grid]
ana_disc = [lebed_analytic(n, frac1=2/3, frac2=1/3) for n in n0_grid]

# Lebed PRD 94, 034039 Table 2 (hard wall, k = sqrt2)
lebed_table2_n0 = [1e-3, 8e-3, 0.125, 1.0, 8.0, 27.0]
lebed_table2_p = [0.1769, 0.1774, 0.1869, 0.2601, 0.5029, 0.5147]

# MC points
rng = np.random.default_rng(1)
mc_n0 = [1.0, 8.0]
mc = {"lebed": [], "discrete": []}
for n0 in mc_n0:
    L = 10.0
    n_each = int(n0 * L**3)
    ps = {"lebed": [], "discrete": []}
    for _ in range(3):
        x = rng.uniform(0, L, (2 * n_each, 3))
        labels = np.concatenate([
            np.repeat(np.arange(3), n_each // 3 + 1)[:n_each],
            np.repeat(np.arange(3, 6), n_each // 3 + 1)[:n_each],
        ]).astype(np.int64)
        for mode in ps:
            p, _ = lebed_static_mc(x, labels, L, mode=mode, rng=rng)
            ps[mode].append(p)
    for mode in ps:
        mc[mode].append((np.mean(ps[mode]),
                         np.std(ps[mode], ddof=1) / np.sqrt(3)))

fig, ax = plt.subplots(figsize=(4.2, 3.2), constrained_layout=True)
ax.semilogx(n0_grid, ana_lebed, color=BLUE, lw=1.8,
            label="Lebed counting (analytic)")
ax.semilogx(n0_grid, ana_disc, color=ORANGE, lw=1.8,
            label="discrete-color counting (analytic)")
ax.plot(lebed_table2_n0, lebed_table2_p, "x", color=INK, ms=7, mew=1.6,
        label="Lebed PRD 94, 034039 (Table 2)", zorder=5)
for mode, col in [("lebed", BLUE), ("discrete", ORANGE)]:
    means = [m for m, _ in mc[mode]]
    errs = [e for _, e in mc[mode]]
    ax.errorbar(mc_n0, means, yerr=errs, fmt="o", color=col, ms=5,
                capsize=3, mfc="white", zorder=6,
                label=f"MC ({mode} mode)")
ax.set_xlabel(r"$n_0 R^3$")
ax.set_ylabel(r"$P(\mathrm{nearest}\ q\ \mathrm{wins};\ k=\sqrt{2})$")
ax.set_ylim(0, 0.6)
ax.legend(fontsize=7.2, frameon=False, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)
fig.savefig("paper/figs/fig2_static.pdf")
fig.savefig("paper/figs/fig2_static.png", dpi=200)
print("saved fig2")
