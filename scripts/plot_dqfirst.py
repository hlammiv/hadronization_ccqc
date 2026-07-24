#!/usr/bin/env python3
"""Fig. 9: diquark-first fraction across scan points vs the static bands."""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from diquark_md.observables import lebed_analytic

BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#333333", "#8a8a8a"

by_point = defaultdict(list)
for f in ["runs/scan_H0.jsonl", "runs/scan_tau.jsonl", "runs/scan_corner.jsonl"]:
    for line in Path(f).read_text().splitlines():
        r = json.loads(line)
        if "error" not in r and r["diquark_first_fraction"] is not None:
            by_point[r["point"]].append(r["diquark_first_fraction"])

points = [("H0_0.05", 0.05, "o", BLUE), ("H0_0.10", 0.10, "o", BLUE),
          ("H0_0.20", 0.20, "o", BLUE), ("corner", 0.05, "s", ORANGE)]

fig, ax = plt.subplots(figsize=(4.2, 3.2), constrained_layout=True)
for name, h0, marker, col in points:
    x = np.array(by_point[name])
    ax.errorbar([h0], [x.mean()], yerr=[x.std(ddof=1) / np.sqrt(len(x))],
                fmt=marker, color=col, ms=6, capsize=3,
                label=None)
ax.plot([], [], "o", color=BLUE, label=r"default ($\Gamma\approx0.8$)")
ax.plot([], [], "s", color=ORANGE, label=r"corner ($\Gamma\approx1.2$)")

# static bands at the simulation density n0 R^3 = 1
p_leb = lebed_analytic(1.0)
p_disc = lebed_analytic(1.0, frac1=2 / 3, frac2=1 / 3)
ax.axhspan(p_leb, p_disc, color=MUTED, alpha=0.25, lw=0)
ax.text(0.21, (p_leb + p_disc) / 2, "static\n(Lebed / discrete)", fontsize=7.5,
        color=INK, ha="right", va="center")

ax.set_xlabel(r"$H_0$ [$c$/fm]")
ax.set_ylabel("diquark-first fraction of baryons")
ax.set_ylim(0, 1.1)
ax.set_xlim(0.02, 0.23)
ax.legend(fontsize=8, frameon=False, loc="center right")
ax.spines[["top", "right"]].set_visible(False)
fig.savefig("paper/figs/fig9_dqfirst.pdf")
fig.savefig("paper/figs/fig9_dqfirst.png", dpi=200)
print("saved fig9")
