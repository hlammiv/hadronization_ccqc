#!/usr/bin/env python3
"""Static Lebed estimator on the INTERACTING equilibrated gas (new result).

Compares P(nearest q wins) on (i) the ideal-gas initial configuration and
(ii) the Langevin/Andersen-equilibrated correlated configuration, in both
counting modes, at the simulation's own density (n_q = 1 fm^-3, R = 1 fm
=> n0 R^3 = 1, where ideal-gas analytics give 26.0% / 31.2%).

Cheap: 3 seeds x (2000-step equilibration of N = 604) ~ 15 s total.
"""

import numpy as np

from diquark_md.io import load_config
from diquark_md.observables import lebed_analytic, lebed_static_mc
from diquark_md.sim import Simulation

R = 1.0
cfg = load_config("configs/default.toml")

rows = {"ideal": {"lebed": [], "discrete": []},
        "equilibrated": {"lebed": [], "discrete": []}}
for seed in range(3):
    sim = Simulation(cfg, seed=100 + seed)
    rng = np.random.default_rng(1000 + seed)
    s = sim.state
    for mode in ("lebed", "discrete"):
        p, n = lebed_static_mc(s["x"], s["labels"], sim.L, R=R, mode=mode, rng=rng)
        rows["ideal"][mode].append(p)
    sim.equilibrate()
    for mode in ("lebed", "discrete"):
        p, n = lebed_static_mc(s["x"], s["labels"], sim.L, R=R, mode=mode, rng=rng)
        rows["equilibrated"][mode].append(p)

n_q = cfg["density"] / 2.0  # per-species density fm^-3
n0R3 = n_q * R**3
print(f"n0 R^3 = {n0R3:.2f}  (T0 = {cfg['t0']} GeV, alpha_s = {cfg['alpha_s']})")
print(f"ideal-gas analytic:  lebed = {lebed_analytic(n0R3):.4f}   "
      f"discrete = {lebed_analytic(n0R3, frac1=2/3, frac2=1/3):.4f}")
for stage in rows:
    line = f"{stage:>13}: "
    for mode in ("lebed", "discrete"):
        x = np.array(rows[stage][mode])
        line += f"{mode} = {x.mean():.4f} +- {x.std(ddof=1)/np.sqrt(len(x)):.4f}   "
    print(line)
