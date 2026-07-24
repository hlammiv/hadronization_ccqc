#!/usr/bin/env python3
"""Fig. 4 (species vs t, paper version) and Fig. 8 (flavor chemistry +
photon spectrum) from run HDF5 files.

Usage:
  python scripts/plot_event.py fig4 runs/default_seed7.h5
  python scripts/plot_event.py fig8 runs/chem_seed11.h5
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
}
INK, MUTED = "#333333", "#8a8a8a"


def load(path):
    with h5py.File(path) as f:
        hist = f["history"][:]
        cols = json.loads(f["history"].attrs["columns"])
        species_hist = json.loads(f.attrs["species_history_json"])
        cfg = json.loads(f.attrs["config_json"])
        photons = f["photon_ledger"][:] if "photon_ledger" in f else np.empty((0, 5))
    return {k: hist[:, i] for i, k in enumerate(cols)}, species_hist, cfg, photons


def fig4(path):
    c, sh, cfg, _ = load(path)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4.2, 4.6), sharex=True,
                                   constrained_layout=True)
    ax1.plot(c["t"], c["t_eff"] * 1e3, color=INK, lw=1.8)
    ax1.axhline(float(cfg["t_chem"]) * 1e3, color=MUTED, lw=1.0, ls="--")
    ax1.text(c["t"][-1], float(cfg["t_chem"]) * 1e3 * 1.1, r"$T_{\rm chem}$",
             ha="right", color=MUTED, fontsize=8)
    ax1.set_ylabel(r"$T_{\rm eff}$ [MeV]")
    ax1.set_yscale("log")
    ts = [r["t"] for r in sh]
    for sp, col in SPECIES_COLORS.items():
        y = np.array([r.get(sp, 0) for r in sh], float)
        if y.max() == 0:
            continue
        ax2.plot(ts, y, color=col, lw=1.7)
        ax2.text(ts[-1], y[-1], f" {sp}", color=col, fontsize=8, va="center")
    ax2.set_xlabel(r"$t$ [fm/$c$]")
    ax2.set_ylabel("bound clusters")
    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig("paper/figs/fig4_species_t.pdf")
    fig.savefig("paper/figs/fig4_species_t.png", dpi=200)
    print("saved fig4")


def fig8(path):
    c, sh, cfg, photons = load(path)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.0),
                                   constrained_layout=True)
    # (a) strangeness fraction + cumulative photons
    s_frac = c["n_s"] / c["n_alive"]
    ax1.plot(c["t"], s_frac, color="#4a3aa7", lw=1.8)
    ax1.set_xlabel(r"$t$ [fm/$c$]")
    ax1.set_ylabel(r"strange fraction $N_s/N$", color="#4a3aa7")
    ax1.set_title("(a) flavor chemistry", fontsize=9, loc="left")
    ax1.spines[["top", "right"]].set_visible(False)

    # (b) escaped-photon (pair) energy spectrum
    if len(photons):
        ax2.hist(photons[:, 1], bins=np.linspace(0, 4, 25),
                 color="#2a78d6", edgecolor="white", lw=0.8)
    ax2.set_xlabel(r"pair energy $E_{\gamma\gamma}$ [GeV]")
    ax2.set_ylabel("annihilations to photons")
    ax2.set_title(f"(b) photon ledger  (n={len(photons)})", fontsize=9,
                  loc="left")
    ax2.spines[["top", "right"]].set_visible(False)
    fig.savefig("paper/figs/fig8_chemistry.pdf")
    fig.savefig("paper/figs/fig8_chemistry.png", dpi=200)
    print("saved fig8")


if __name__ == "__main__":
    {"fig4": fig4, "fig8": fig8}[sys.argv[1]](sys.argv[2])
