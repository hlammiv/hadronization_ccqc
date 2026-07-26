#!/usr/bin/env python3
"""Size-scan figure: composite-to-meson ratios vs quark multiplicity."""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FILES = ["runs/scan_smallsys.jsonl", "runs/scan_smallsys_hi.jsonl",
         "runs/scan_smallsys_topup.jsonl", "runs/scan_n1000.jsonl",
         "runs/prod_H0.jsonl", "runs/scan_H0_more.jsonl"]
POINTS = [("N96_H0.1", 96), ("N300_H0.1", 300), ("N1000_H0.1", 1000),
          ("H0_0.10", 3000)]
ORANGE, YELLOW = "#eb6834", "#eda100"

by_point = defaultdict(list)
for f in FILES:
    for line in Path(f).read_text().splitlines():
        r = json.loads(line)
        if "error" not in r:
            by_point[r["point"]].append(r)


def ratio(point, sps):
    recs = by_point[point]
    m = np.array([r["species_counts"].get("meson", 0) for r in recs], float)
    s = np.array([np.mean([r["species_counts"].get(sp, 0) for sp in sps])
                  for r in recs], float)
    rng = np.random.default_rng(3)
    idx = rng.integers(0, len(recs), (2000, len(recs)))
    boots = s[idx].mean(axis=1) / m[idx].mean(axis=1)
    return s.mean() / m.mean(), boots.std(ddof=1), s.sum(), len(recs)


fig, ax = plt.subplots(figsize=(4.2, 3.2), constrained_layout=True)
for sps, col, lab in [(("baryon", "antibaryon"), ORANGE, "baryon / meson"),
                      (("tetraquark",), YELLOW, "tetraquark / meson")]:
    xs, ys, es = [], [], []
    for name, nq in POINTS:
        r, e, tot, nev = ratio(name, sps)
        if tot > 0:
            xs.append(nq)
            ys.append(r)
            es.append(e)
        else:
            recs = by_point[name]
            m_mean = np.mean([rr["species_counts"].get("meson", 0)
                              for rr in recs])
            ul = 2.996 / nev / m_mean
            ax.errorbar([nq], [ul], yerr=[[ul * 0.6], [0]], uplims=[True],
                        fmt="_", color=col, ms=10)
        print(f"{lab} {name} (n={nev}): "
              f"{r*100:.2f} +- {e*100:.2f} %" if tot > 0 else
              f"{lab} {name} (n={nev}): UL")
    ax.errorbar(xs, ys, yerr=es, fmt="o-", color=col, lw=1.6, ms=5,
                capsize=3, label=lab)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"system size $N_{\rm quarks}$")
ax.set_ylabel("yield ratio to mesons")
ax.legend(fontsize=8, frameon=False, loc="lower right")
ax.spines[["top", "right"]].set_visible(False)
fig.savefig("paper/figs/fig10_smallsys.pdf")
fig.savefig("paper/figs/fig10_smallsys.png", dpi=200)
print("saved fig10")
