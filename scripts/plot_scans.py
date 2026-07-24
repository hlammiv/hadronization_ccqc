#!/usr/bin/env python3
"""Aggregate scan JSONL results -> ratio tables + scan figures.

Usage: python scripts/plot_scans.py runs/scan_H0.jsonl runs/scan_tau.jsonl ...
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SPECIES_COLORS = {
    "meson": "#2a78d6",
    "baryon": "#eb6834",
    "antibaryon": "#1baf7a",
    "tetraquark": "#eda100",
    "pentaquark": "#e87ba4",
    "antipentaquark": "#e87ba4",
}
ORDER = ["meson", "baryon", "antibaryon", "tetraquark", "pentaquark",
         "antipentaquark"]


def load(paths):
    by_point = defaultdict(list)
    for path in paths:
        for line in Path(path).read_text().splitlines():
            rec = json.loads(line)
            if "error" in rec:
                print("SKIP (error):", rec["point"], rec["seed"], rec["error"])
                continue
            by_point[rec["point"]].append(rec)
    return by_point


def per_event_counts(recs, sp):
    return np.array([r["species_counts"].get(sp, 0) for r in recs], float)


def bootstrap_mean(x, n_boot=2000, rng=None):
    rng = rng or np.random.default_rng(0)
    if len(x) == 0:
        return 0.0, 0.0
    means = rng.choice(x, (n_boot, len(x))).mean(axis=1)
    return float(x.mean()), float(means.std(ddof=1))


def poisson_upper95(total_count, n_events):
    """Garwood/exact 95% CL upper limit on the per-event mean."""
    from math import inf

    try:
        from scipy.stats import chi2  # optional

        ul = chi2.ppf(0.95, 2 * (total_count + 1)) / 2.0
    except ImportError:
        # Stirling-free small-count table for k = 0..5 (exact values)
        table = {0: 2.996, 1: 4.744, 2: 6.296, 3: 7.754, 4: 9.154, 5: 10.51}
        ul = table.get(total_count, inf)
    return ul / n_events


def summarize(by_point):
    rows = {}
    rng = np.random.default_rng(1)
    for point, recs in sorted(by_point.items()):
        row = {"n_events": len(recs)}
        for sp in ORDER:
            x = per_event_counts(recs, sp)
            row[sp] = bootstrap_mean(x, rng=rng)
            row[f"{sp}_total"] = int(x.sum())
        # rare-species upper limits
        for sp in ("tetraquark", "pentaquark"):
            row[f"{sp}_ul95"] = poisson_upper95(row[f"{sp}_total"], len(recs))
        dq = [r["diquark_first_fraction"] for r in recs
              if r["diquark_first_fraction"] is not None]
        nb = sum(r["n_pathway_baryons"] for r in recs)
        row["dq_frac"] = (float(np.mean(dq)) if dq else float("nan"),
                          float(np.std(dq, ddof=1) / np.sqrt(len(dq)))
                          if len(dq) > 1 else 0.0)
        row["n_pathway_baryons"] = nb
        row["photons"] = sum(r["n_photons"] for r in recs)
        rows[point] = row
    return rows


def print_table(rows):
    hdr = (f"{'point':<12}{'ev':>4} " +
           "".join(f"{sp:>16}" for sp in ORDER) +
           f"{'dq-first':>14}{'photons':>9}")
    print(hdr)
    for point, row in rows.items():
        cells = ""
        for sp in ORDER:
            m, e = row[sp]
            cells += f"{m:8.3f}+-{e:5.3f} "
        dqm, dqe = row["dq_frac"]
        print(f"{point:<12}{row['n_events']:>4} {cells}"
              f"{dqm:7.2f}+-{dqe:4.2f} {row['photons']:>8}")
        for sp in ("tetraquark", "pentaquark"):
            if row[f"{sp}_total"] == 0:
                print(f"{'':<17}{sp}: 0 observed in {row['n_events']} events "
                      f"-> <{row[f'{sp}_ul95']:.3f}/event (95% CL)")


def scan_figure(rows, x_map, xlabel, out, logx=False):
    points = [p for p in rows if p in x_map]
    xs = np.array([x_map[p] for p in points])
    order = np.argsort(xs)
    xs = xs[order]
    points = [points[i] for i in order]

    fig, ax = plt.subplots(figsize=(4.4, 3.3), constrained_layout=True)
    for sp in ORDER:
        ys = np.array([rows[p][sp][0] for p in points])
        es = np.array([rows[p][sp][1] for p in points])
        if ys.max() <= 0:
            continue
        ax.errorbar(xs, ys, yerr=es, marker="o", ms=4.5, lw=1.6, capsize=3,
                    color=SPECIES_COLORS[sp], label=sp)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("yield per event")
    ax.set_yscale("log")
    if logx:
        ax.set_xscale("log")
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(out)
    fig.savefig(str(Path(out).with_suffix(".png")), dpi=200)
    print("saved", out)


if __name__ == "__main__":
    by_point = load(sys.argv[1:])
    rows = summarize(by_point)
    print_table(rows)

    h0_map = {"H0_0.05": 0.05, "H0_0.10": 0.10, "H0_0.20": 0.20}
    if any(p in rows for p in h0_map):
        scan_figure(rows, h0_map, r"$H_0$ [$c$/fm]",
                    "paper/figs/fig5_scan_H0.pdf")
    tau_map = {"tau_5": 5.0, "tau_20": 20.0, "tau_50": 50.0, "tau_inf": 200.0}
    if any(p in rows for p in tau_map):
        scan_figure(rows, tau_map,
                    r"$\tau_{\rm bound}$ [fm/$c$]  (rightmost: $\infty$)",
                    "paper/figs/fig7_scan_tau.pdf", logx=True)
