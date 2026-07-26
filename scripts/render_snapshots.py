#!/usr/bin/env python3
"""Fig. 3: three-panel event render with an external zoom column.

Full-box x-y projection per time slice: unbound quarks are faint points;
members of eventual clusters are colored by species and bonded once the
cluster has assembled (all pair separations within 1.2 lambda_D(t), which
grows with the comoving screening).  Zoom boxes live in a dedicated fourth
column OUTSIDE the final panel, so they can never occlude data or each
other; their targets are chosen from the right half of the panel and the
slots are ordered by target height, keeping connectors short and parallel.

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
RARE = {"baryon", "antibaryon", "tetraquark", "pentaquark", "antipentaquark"}
FREE = "#c9c9c9"
ABBREV = {"meson": "M", "baryon": "B", "antibaryon": r"$\bar{\rm B}$",
          "tetraquark": "T", "pentaquark": "P",
          "antipentaquark": r"$\bar{\rm P}$"}


def bond_cut(lam_d0, h0, t):
    """Assembled once all pairs are within ~1.2 lambda_D(t)."""
    return 1.2 * lam_d0 * (1.0 + h0 * t)


def recenter(pts, L):
    ref = pts[0]
    d = pts - ref
    d -= L * np.round(d / L)
    return ref + d


def max_sep(pts):
    return max(np.linalg.norm(pts[a] - pts[b])
               for a in range(len(pts)) for b in range(a + 1, len(pts)))


def main(path):
    with h5py.File(path) as f:
        cfg = json.loads(f.attrs["config_json"])
        lam_d0, h0 = float(cfg["lam_d0"]), float(cfg["h0"])
        keys = sorted(f["snapshots"].keys())
        picks = [keys[0], keys[len(keys) // 2], keys[-1]]
        snaps = []
        for k in picks:
            g = f[f"snapshots/{k}"]
            snaps.append((g.attrs["t"], g.attrs["L"], g["x"][:], g["index"][:]))
        members = json.loads(f.attrs["final_clusters_json"])

    clusters = [(tuple(map(int, mstr.split(","))), species)
                for mstr, species in members]

    fig = plt.figure(figsize=(11.4, 3.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 0.36])
    axes = [fig.add_subplot(gs[:, i]) for i in range(3)]
    zoom_axes = [fig.add_subplot(gs[0, 3]), fig.add_subplot(gs[1, 3])]

    for ax, (t, L, x, idx) in zip(axes, snaps):
        pos = {int(i): x[k] for k, i in enumerate(idx)}
        in_cluster = {i for mem, _ in clusters for i in mem}
        free_pts = np.array([pos[i] for i in pos if i not in in_cluster])
        ax.scatter(free_pts[:, 0], free_pts[:, 1], s=2.5, c=FREE,
                   linewidths=0, alpha=0.45)

        counts = {}
        for mem, species in clusters:
            col = SPECIES_COLORS.get(species, "#008300")
            pts = np.array([pos[i] for i in mem if i in pos])
            if len(pts) == 0:
                continue
            pts = recenter(pts, L)
            if max_sep(pts) >= bond_cut(lam_d0, h0, t):
                ax.scatter(pts[:, 0] % L, pts[:, 1] % L, s=2.5, c=FREE,
                           linewidths=0, alpha=0.45)
                continue
            for a in range(len(pts)):
                for b in range(a + 1, len(pts)):
                    ax.plot(pts[[a, b], 0], pts[[a, b], 1],
                            color=col, lw=1.1, alpha=0.9, zorder=3)
            big = species in RARE
            ax.scatter(pts[:, 0], pts[:, 1], s=(60 if big else 9), c=col,
                       linewidths=(0.9 if big else 0.3),
                       edgecolors="white", zorder=(5 if big else 4))
            counts[species] = counts.get(species, 0) + 1

        ax.set_xlim(0, L)
        ax.set_ylim(0, L)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        order = ["meson", "baryon", "antibaryon", "tetraquark",
                 "pentaquark", "antipentaquark"]
        label = "  ".join(f"{counts[k]} {ABBREV[k]}" for k in order
                          if k in counts)
        ax.set_title(f"$t={t:.0f}$ fm/$c$,  $L={L:.0f}$ fm:  {label}",
                     fontsize=8, loc="left")

    # ---- external zoom column, fed from the final panel ----
    t, L, x, idx = snaps[-1]
    pos = {int(i): x[k] for k, i in enumerate(idx)}
    ax3 = axes[2]
    cut_f = bond_cut(lam_d0, h0, t)

    def geom(mem):
        pts = recenter(np.array([pos[i] for i in mem if i in pos]), L)
        return pts, (pts.mean(axis=0) % L) / L, max_sep(pts)

    compact = []
    for mem, sp in clusters:
        present = [i for i in mem if i in pos]
        if len(present) < 2:
            continue
        pts, frac, sep = geom(mem)
        if sep < cut_f:
            compact.append((mem, sp, pts, frac, sep))
    exotics = [c for c in compact if len(c[0]) >= 4]
    baryons = [c for c in compact if c[1] in ("baryon", "antibaryon")]

    chosen = []
    if exotics:
        top = max(len(c[0]) for c in exotics)
        pool = [c for c in exotics if len(c[0]) == top]
        chosen.append(max(pool, key=lambda c: c[3][0]))   # rightmost target
    if baryons:
        chosen.append(max(baryons, key=lambda c: c[3][0]))
    # top slot serves the higher target; connectors stay parallel
    chosen.sort(key=lambda c: -c[3][1])

    for (mem, species, pts, frac, sep), zax in zip(chosen, zoom_axes):
        col = SPECIES_COLORS.get(species, "#008300")
        w = 0.75 * sep + 0.5
        zax.scatter(pts[:, 0], pts[:, 1], s=110, c=col, edgecolors="white",
                    linewidths=1.1, zorder=5)
        for a in range(len(pts)):
            for b in range(a + 1, len(pts)):
                zax.plot(pts[[a, b], 0], pts[[a, b], 1], color=col, lw=1.8)
        cc = pts.mean(axis=0)
        zax.set_xlim(cc[0] - w, cc[0] + w)
        zax.set_ylim(cc[1] - w, cc[1] + w)
        zax.set_xticks([])
        zax.set_yticks([])
        for s in zax.spines.values():
            s.set_color(col)
            s.set_linewidth(1.5)
        zax.set_title(species, fontsize=8, color=col, pad=2)
        ax3.indicate_inset(
            bounds=[cc[0] - w, cc[1] - w, 2 * w, 2 * w],
            inset_ax=zax, edgecolor=col, lw=1.2, alpha=0.9,
        )
    for zax in zoom_axes[len(chosen):]:
        zax.axis("off")

    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=sp,
                          markersize=6)
               for sp, c in list(SPECIES_COLORS.items())[:5]]
    handles.append(plt.Line2D([], [], marker="o", ls="", color=FREE,
                              label="unbound", markersize=4))
    axes[0].legend(handles=handles, fontsize=6.4, frameon=False,
                   loc="lower left")
    fig.savefig("paper/figs/fig3_render.pdf")
    fig.savefig("paper/figs/fig3_render.png", dpi=200)
    print("saved fig3")


if __name__ == "__main__":
    main(sys.argv[1])
