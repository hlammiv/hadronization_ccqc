#!/usr/bin/env python3
"""Fig. 3: three-panel event render with cluster bonds and zoom insets.

Full-box x-y projection: unbound quarks are faint points; members of
eventual clusters are colored by species with bonds drawn once members
approach each other (assembly becomes visible); rare species get large
ringed markers; the final panel carries zoom insets on a baryon and on
the largest exotic cluster.

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
# a cluster counts as assembled once all pairs are within ~1.2 lambda_D(t):
# late captures are wide (comoving screening), so the cut must scale with a(t)
def bond_cut(lam_d0, h0, t):
    return 1.2 * lam_d0 * (1.0 + h0 * t)
ABBREV = {"meson": "M", "baryon": "B", "antibaryon": r"$\bar{\rm B}$",
          "tetraquark": "T", "pentaquark": "P", "antipentaquark": r"$\bar{\rm P}$"}


def recenter(pts, L):
    """Minimum-image recentering of cluster members about the first one."""
    ref = pts[0]
    d = pts - ref
    d -= L * np.round(d / L)
    return ref + d


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

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.6),
                             constrained_layout=True)
    for panel, (ax, (t, L, x, idx)) in enumerate(zip(axes, snaps)):
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
            sep = max(np.linalg.norm(pts[a] - pts[b])
                      for a in range(len(pts)) for b in range(a + 1, len(pts)))
            assembled = sep < bond_cut(lam_d0, h0, t)
            if not assembled:
                # not yet a hadron: indistinguishable from the free gas
                ax.scatter(pts[:, 0] % L, pts[:, 1] % L, s=2.5, c=FREE,
                           linewidths=0, alpha=0.45)
                continue
            for a in range(len(pts)):
                for b in range(a + 1, len(pts)):
                    ax.plot(pts[[a, b], 0], pts[[a, b], 1],
                            color=col, lw=1.1, alpha=0.9, zorder=3)
            big = species in RARE
            ax.scatter(pts[:, 0], pts[:, 1],
                       s=(60 if big else 9), c=col,
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

    # zoom insets on the final panel: one baryon, one largest exotic
    t, L, x, idx = snaps[-1]
    pos = {int(i): x[k] for k, i in enumerate(idx)}
    ax = axes[2]

    def add_inset(mem, species, loc):
        pts = np.array([pos[i] for i in mem if i in pos])
        if len(pts) == 0:
            return
        pts = recenter(pts, snaps[-1][1])
        c = pts.mean(axis=0)
        sep = max(np.linalg.norm(pts[a] - pts[b])
                  for a in range(len(pts)) for b in range(a + 1, len(pts)))
        w = 0.75 * sep + 0.5
        col = SPECIES_COLORS.get(species, "#008300")
        axi = ax.inset_axes(loc)
        axi.scatter(pts[:, 0], pts[:, 1], s=90, c=col, edgecolors="white",
                    linewidths=1.0, zorder=5)
        for a in range(len(pts)):
            for b in range(a + 1, len(pts)):
                axi.plot(pts[[a, b], 0], pts[[a, b], 1], color=col, lw=1.6)
        axi.set_xlim(c[0] - w, c[0] + w)
        axi.set_ylim(c[1] - w, c[1] + w)
        axi.set_xticks([]); axi.set_yticks([])
        for s in axi.spines.values():
            s.set_color(col); s.set_linewidth(1.4)
        axi.set_title(species, fontsize=7, color=col, pad=1.5)
        ax.indicate_inset_zoom(axi, edgecolor=col, lw=1.0)

    def final_sep(mem):
        pts = recenter(np.array([pos[i] for i in mem if i in pos]),
                       snaps[-1][1])
        return max(np.linalg.norm(pts[a] - pts[b])
                   for a in range(len(pts)) for b in range(a + 1, len(pts)))

    cut_f = bond_cut(lam_d0, h0, snaps[-1][0])
    baryons = sorted(((m, s) for m, s in clusters
                      if s in ("baryon", "antibaryon") and final_sep(m) < cut_f),
                     key=lambda c: final_sep(c[0]))
    exotics = sorted((c for c in clusters
                      if len(c[0]) >= 4 and final_sep(c[0]) < cut_f),
                     key=lambda c: (-len(c[0]), final_sep(c[0])))
    if baryons:
        add_inset(*baryons[0], loc=[0.02, 0.66, 0.30, 0.30])
    if exotics:
        add_inset(*exotics[0], loc=[0.66, 0.02, 0.30, 0.30])

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
