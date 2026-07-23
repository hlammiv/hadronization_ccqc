"""Observables: static Lebed nearest-neighbor estimator, energy audit, T_eff.

Static estimator conventions (locked against Lebed PRD 94, 034039):

Hard-wall screening radius R.  Both nearest-neighbor distributions are
truncated at R and normalized (i.e., conditioned on a nearest attractive
quark AND a nearest singlet-channel antiquark existing within R).  The
diquark "wins" iff r_qbar > k r_q with k = sqrt(2) (inverse-square force
comparison with the 2:1 coupling ratio).

Effective partner densities:
  - "lebed" mode:    n_1 = n_q / 3 (3bar fraction of qq),  n_2 = n_qbar / 9
                     (singlet fraction of q qbar), full-strength channels.
  - "discrete" mode: n_1 = 2 n_q / 3 (different-color quarks),
                     n_2 = n_qbar / 3 (matched-anticolor antiquarks); the
                     mean-channel coupling ratio is still 2:1 so k = sqrt(2).

Verified limits: n0 -> 0 gives 17.68% and n0 = 1/R^3 gives 26.0%
(Lebed Table 2); n0 -> inf gives n_1/(k^3 n_2 + n_1) = 51.47%.
"""

import numpy as np

from .integrator import effective_temperature, kinetic_energy  # noqa: F401 (re-export)


def lebed_analytic(n0, R=1.0, k=np.sqrt(2.0), frac1=1.0 / 3.0, frac2=1.0 / 9.0,
                   n_grid=20000):
    """Analytic hard-wall probability that the nearest attractive quark beats
    the nearest singlet antiquark, at per-species density n0.

    n_1 = frac1 * n0, n_2 = frac2 * n0.
    """
    n1 = frac1 * n0
    n2 = frac2 * n0
    a1 = (4.0 * np.pi / 3.0) * n1
    a2 = (4.0 * np.pi / 3.0) * n2
    r = np.linspace(0.0, R / k, n_grid)
    w1 = 4.0 * np.pi * r**2 * n1 * np.exp(-a1 * r**3)
    s2_kr = np.exp(-a2 * (k * r) ** 3)
    s2_R = np.exp(-a2 * R**3)
    num = np.trapezoid(w1 * (s2_kr - s2_R), r)
    norm1 = 1.0 - np.exp(-a1 * R**3)
    norm2 = 1.0 - s2_R
    return num / (norm1 * norm2)


def _nearest_in_set(x_test, x_set, L):
    """Minimum-image nearest distance from x_test to each point in x_set."""
    if len(x_set) == 0:
        return np.inf
    d = x_set - x_test
    d -= L * np.round(d / L)
    return np.sqrt((d**2).sum(axis=1).min())


def lebed_static_mc(x, labels, L, R=1.0, k=np.sqrt(2.0), mode="lebed", rng=None):
    """Monte-Carlo static diquark-formation probability on a configuration.

    Conditioned on both partners within R (matching `lebed_analytic`).
    Returns (probability, n_events_conditioned).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    labels = np.asarray(labels)
    quarks = np.where(labels < 3)[0]
    antiquarks = np.where(labels >= 3)[0]

    wins = 0
    events = 0
    for i in quarks:
        others_q = quarks[quarks != i]
        if mode == "lebed":
            # each qq pair is 3bar (full strength) w.p. 1/3;
            # each q qbar pair is singlet (full strength) w.p. 1/9
            att = others_q[rng.random(len(others_q)) < (1.0 / 3.0)]
            sing = antiquarks[rng.random(len(antiquarks)) < (1.0 / 9.0)]
        elif mode == "discrete":
            # attractive quarks: different color (2/3); singlet-content
            # antiquarks: matched anticolor (1/3); strengths in 2:1 ratio
            att = others_q[labels[others_q] != labels[i]]
            sing = antiquarks[labels[antiquarks] == labels[i] + 3]
        else:
            raise ValueError(f"unknown mode {mode!r}")
        r1 = _nearest_in_set(x[i], x[att], L)
        r2 = _nearest_in_set(x[i], x[sing], L)
        if r1 < R and r2 < R:
            events += 1
            if r2 > k * r1:
                wins += 1
    return (wins / events if events else np.nan), events


def energy_audit(x, p, labels, mass, alive, c_table, L, alpha_s, lam_d, r0, r_cut,
                 ledger_energy=0.0):
    """Total energy bookkeeping: kinetic + potential + escaped/ledger."""
    from .potentials import compute_forces

    _, pot = compute_forces(x, labels, alive, c_table, L, alpha_s, lam_d, r0, r_cut)
    kin = kinetic_energy(p, mass, alive)
    rest = float(np.sum(mass[alive]))
    return {
        "kinetic": kin,
        "potential": pot,
        "rest_mass": rest,
        "ledger": ledger_energy,
        "total": kin + pot + rest + ledger_energy,
    }
