#!/usr/bin/env python3
"""(Gamma, kappa) heatmap, stacked single-column layout."""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from diquark_md.units import coupling_gamma, screening_kappa

FILL = {"a0.5_l0.6", "a0.6_l0.3", "a0.6_l0.4", "a0.6_l0.6"}
by_point = defaultdict(list)
for f, skip_fill in [("runs/scan_heatmap.jsonl", True),
                     ("runs/scan_heatmap_fill.jsonl", False)]:
    for line in Path(f).read_text().splitlines():
        r = json.loads(line)
        if "error" in r:
            continue
        if skip_fill and r["point"] in FILL:
            continue
        by_point[r["point"]].append(r)

alphas = [0.3, 0.4, 0.5, 0.6]
lams = [0.3, 0.4, 0.6]
BM = np.zeros((3, 4))
TM = np.zeros_like(BM)
for i, lam in enumerate(lams):
    for j, a in enumerate(alphas):
        recs = by_point[f"a{a}_l{lam}"]
        assert len(recs) == 16, (a, lam, len(recs))
        m = np.mean([r["species_counts"].get("meson", 0) for r in recs])
        b = np.mean([(r["species_counts"].get("baryon", 0)
                      + r["species_counts"].get("antibaryon", 0)) / 2
                     for r in recs])
        t = np.mean([r["species_counts"].get("tetraquark", 0) for r in recs])
        BM[i, j] = b / m * 100
        TM[i, j] = t / m * 100

gammas = [coupling_gamma(a, 2.0, 0.2) for a in alphas]
kappas = [screening_kappa(2.0, l) for l in lams]

fig, axes = plt.subplots(2, 1, figsize=(3.5, 5.2), constrained_layout=True)
for ax, Z, title in [(axes[0], BM, "(a) baryon/meson [%]"),
                     (axes[1], TM, "(b) tetraquark/meson [%]")]:
    ax.imshow(Z, origin="lower", aspect="auto", cmap="Blues")
    ax.set_xticks(range(4), [f"{g:.1f}" for g in gammas])
    ax.set_yticks(range(3), [f"{k:.2f}" for k in kappas])
    ax.set_xlabel(r"$\Gamma$")
    ax.set_ylabel(r"$\kappa$")
    ax.set_title(title, fontsize=9, loc="left")
    for i in range(3):
        for j in range(4):
            ax.text(j, i, f"{Z[i, j]:.2f}", ha="center", va="center",
                    fontsize=7.5,
                    color="white" if Z[i, j] > 0.6 * Z.max() else "#333333")
fig.savefig("paper/figs/fig11_heatmap.pdf")
fig.savefig("paper/figs/fig11_heatmap.png", dpi=200)
print("saved fig11 (stacked)")
