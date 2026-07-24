#!/usr/bin/env python3
"""Bound-meson lifetime calibration: measured decay time vs tau_bound.

Prepares an isolated bound matched q-qbar pair (orbit apoapsis beyond
d_ann), evolves with reactions at P_gamma = 1, and records the decay
time.  The effective lifetime is tau_eff = 3 tau_bound / duty, where
duty is the orbit's fractional time inside d_ann; the fitted slope
tau_eff / tau_bound gives the duty-cycle factor quoted in the paper.

Cheap: 2-particle dynamics; ~200 trials x 3 tau values ~ seconds.
"""

import math

import numpy as np

from diquark_md import color
from diquark_md.integrator import drift, kick
from diquark_md.potentials import compute_forces
from diquark_md.reactions import ReactionEngine

C_TABLE = color.build_c_table()
ALPHA, LAM_D, R0, D_ANN = 0.4, 0.4, 0.2, 0.2
L = 50.0
DT = 0.005
T_MAX = 400.0


def one_trial(tau_bound, seed):
    rng = np.random.default_rng(seed)
    # bound orbit: start at r = 0.3 fm with small tangential momentum
    x = np.array([[25.0, 25.0, 25.0], [25.3, 25.0, 25.0]])
    p = np.array([[0.0, 0.015, 0.0], [0.0, -0.015, 0.0]])
    state = dict(x=x, p=p,
                 labels=np.array([0, 3], dtype=np.int64),
                 flavors=np.array([0, 0], dtype=np.int64),
                 mass=np.array([0.34, 0.34]),
                 alive=np.ones(2, dtype=np.bool_))
    eng = ReactionEngine(dict(lambda_scatt=0.0, tau_bound=tau_bound,
                              p_gamma=1.0, t_chem=0.16, d_ann=D_ANN), rng)
    pot = (C_TABLE, ALPHA, LAM_D, R0, min(L / 2, 6 * LAM_D), L)
    forces, _ = compute_forces(state["x"], state["labels"], state["alive"],
                               C_TABLE, L, ALPHA, LAM_D, R0, pot[4])
    n_steps = int(T_MAX / DT)
    inside = 0
    for step in range(n_steps):
        kick(state["p"], forces, state["alive"], 0.5 * DT)
        drift(state["x"], state["p"], state["mass"], state["alive"], DT, L)
        forces, _ = compute_forces(state["x"], state["labels"], state["alive"],
                                   C_TABLE, L, ALPHA, LAM_D, R0, pot[4])
        kick(state["p"], forces, state["alive"], 0.5 * DT)
        d = state["x"][0] - state["x"][1]
        d -= L * np.round(d / L)
        if np.linalg.norm(d) < D_ANN:
            inside += 1
        if eng.step(state, step * DT, DT, t_eff=0.01, potential_pars=pot):
            return step * DT, inside / (step + 1)
    return None, inside / n_steps


if __name__ == "__main__":
    # duty cycle from a long reaction-free orbit (tau = inf)
    _, duty = one_trial(math.inf, seed=0)
    print(f"orbit duty cycle inside d_ann: {duty:.3f}  "
          f"(predicted tau_eff = 3 tau / {duty:.3f} = {3.0/duty:.1f} tau)")

    n_trials = 150
    for tau in (0.5, 1.0, 2.0):
        times = []
        censored = 0
        for k in range(n_trials):
            t_dec, _ = one_trial(tau, seed=10_000 + k)
            if t_dec is None:
                censored += 1
            else:
                times.append(t_dec)
        times = np.array(times)
        # exponential MLE with right-censoring at T_MAX
        total_time = times.sum() + censored * T_MAX
        tau_eff = total_time / max(len(times), 1)
        err = tau_eff / math.sqrt(max(len(times), 1))
        print(f"tau_bound = {tau:4.1f}: tau_eff = {tau_eff:6.1f} +- {err:4.1f} "
              f"fm/c   ratio = {tau_eff/tau:5.1f}  "
              f"(decayed {len(times)}/{n_trials})")
