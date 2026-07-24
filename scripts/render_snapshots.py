#!/usr/bin/env python3
"""Fig. 3: three-panel event render, particles colored by eventual species.

Usage: python scripts/render_snapshots.py runs/render_event.h5
"""

import json
import sys

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SPECIES_COLORS = {
    "meson": "#2a78d6", "baryon": "#eb6834", "antibaryon": "#1baf7a",
    "tetraquark": "#eda100", "pentaquark": "#e87ba4",
    "antipentaquark": "#e87ba4",
}
FREE = "#c9c9c9"


def main(path):
    with h5py.File(path) as f:
        keys = sorted(f["snapshots"].keys())
        # early / mid / late
        picks = [keys[0], keys[len(keys) // 2], keys[-1]]
        snaps = []
        for k in picks:
            g = f[f"snapshots/{k}"]
            snaps.append((g.attrs["t"], g.attrs["L"], g["x"][:], g["index"][:]))
        members = json.loads(f.attrs["final_clusters_json"])

    color_of = {}
    for mstr, species in members:
        for i in map(int, mstr.split(",")):
            color_of[i] = SPECIES_COLORS.get(species, "#008300")

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.4), constrained_layout=True)
    for ax, (t, L, x, idx) in zip(axes, snaps):
        cols = [color_of.get(int(i), FREE) for i in idx]
        bound = np.array([c != FREE for c in cols])
        # slab projection: keep |z - L/2| < L/6 for legibility
        sel = np.abs(x[:, 2] - L / 2) < L / 6
        ax.scatter(x[sel & ~bound, 0], x[sel & ~bound, 1], s=4, c=FREE,
                   linewidths=0, alpha=0.6)
        ax.scatter(x[sel & bound, 0], x[sel & bound, 1], s=10,
                   c=[c for c, s_, b in zip(cols, sel, bound) if s_ and b],
                   linewidths=0)
        ax.set_xlim(0, L)
        ax.set_ylim(0, L)
        ax.set_aspect("equal")
        ax.set_title(f"$t = {t:.0f}$ fm/$c$   ($L = {L:.0f}$ fm)",
                     fontsize=9, loc="left")
        ax.set_xticks([])
        ax.set_yticks([])
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=sp)
               for sp, c in list(SPECIES_COLORS.items())[:5]]
    handles.append(plt.Line2D([], [], marker="o", ls="", color=FREE,
                              label="unbound"))
    axes[2].legend(handles=handles, fontsize=6.5, frameon=False,
                   loc="upper right")
    fig.savefig("paper/figs/fig3_render.pdf")
    fig.savefig("paper/figs/fig3_render.png", dpi=200)
    print("saved fig3")


if __name__ == "__main__":
    main(sys.argv[1])
