#!/usr/bin/env python3
"""Species-formation history figure from a diquark_md HDF5 run.

Usage: python scripts/plot_species.py runs/default_seed7.h5 [out.pdf]
"""

import json
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# validated categorical palette (dataviz reference instance, fixed order;
# color follows the species, never its rank)
SPECIES_COLORS = {
    "meson": "#2a78d6",
    "baryon": "#eb6834",
    "antibaryon": "#1baf7a",
    "tetraquark": "#eda100",
    "pentaquark": "#e87ba4",
    "antipentaquark": "#e87ba4",
    "other": "#008300",
}
INK = "#333333"
MUTED = "#8a8a8a"


def main(path, out=None):
    with h5py.File(path) as f:
        hist = f["history"][:]
        cols = json.loads(f["history"].attrs["columns"])
        species_hist = json.loads(f.attrs["species_history_json"])
        final = json.loads(f["final"].attrs["species_counts"])
        exotic = json.loads(f["final"].attrs["exotic_counts"])
        dq = f["final"].attrs["diquark_first_fraction"]
        cfg = json.loads(f.attrs["config_json"])

    c = {k: hist[:, i] for i, k in enumerate(cols)}
    all_species = sorted(
        {k for rec in species_hist for k in rec if k != "t"},
        key=lambda s: (s not in SPECIES_COLORS, s),
    )

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), constrained_layout=True)

    # (a) thermodynamic history
    ax = axes[0]
    ax.plot(c["t"], c["t_eff"] * 1e3, color=INK, lw=1.8)
    ax.axhline(float(cfg["t_chem"]) * 1e3, color=MUTED, lw=1.0, ls="--")
    ax.text(c["t"][-1], float(cfg["t_chem"]) * 1e3, " $T_{\\rm chem}$",
            va="bottom", ha="right", color=MUTED, fontsize=9)
    ax.set_xlabel("$t$ [fm/$c$]")
    ax.set_ylabel("$T_{\\rm eff}$ [MeV]")
    ax.set_yscale("log")
    ax.set_title("(a) cooling history", fontsize=10, loc="left")

    # (b) species counts vs time
    ax = axes[1]
    t_vals = [rec["t"] for rec in species_hist]
    for sp in all_species:
        y = np.array([rec.get(sp, 0) for rec in species_hist], dtype=float)
        if y.max() == 0:
            continue
        col = SPECIES_COLORS.get(sp, SPECIES_COLORS["other"])
        ax.plot(t_vals, y, color=col, lw=1.8)
        ax.text(t_vals[-1], y[-1], f" {sp}", color=col, fontsize=9,
                va="center")
    ax.set_xlabel("$t$ [fm/$c$]")
    ax.set_ylabel("bound clusters (per snapshot)")
    ax.set_title("(b) species formation", fontsize=10, loc="left")

    # (c) final persistent tally
    ax = axes[2]
    if final:
        names = sorted(final, key=lambda s: -final[s])
        vals = [final[s] for s in names]
        cols_ = [SPECIES_COLORS.get(s, SPECIES_COLORS["other"]) for s in names]
        bars = ax.bar(range(len(names)), vals, color=cols_, width=0.62)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, str(v), ha="center",
                    va="bottom", color=INK, fontsize=9)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("persistent clusters")
    subtitle = "(c) final tally"
    if not np.isnan(dq) and dq >= 0:
        subtitle += f"  (diquark-first: {dq:.2f})"
    if exotic:
        subtitle += f"  exotic: {exotic}"
    ax.set_title(subtitle, fontsize=10, loc="left")

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors=INK, labelsize=9)

    out = out or str(Path(path).with_suffix(".pdf"))
    fig.savefig(out, dpi=200)
    png = str(Path(out).with_suffix(".png"))
    fig.savefig(png, dpi=200)
    print(f"saved {out} and {png}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
