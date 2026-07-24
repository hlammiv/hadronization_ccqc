"""Poisson-hazard annihilation with photon/gluon branching (plan Rev. 2).

Eligible pairs: matching-color, same-flavor q-qbar within d_ann.  Hazards
(per unit time, singlet weight 1/3 folded in):

    unbound pair (E_pair > 0):  lambda_scatt / 3
    bound pair   (E_pair < 0):  1 / (3 tau_bound)     [mesons and q-qbar
                                 sub-pairs inside larger clusters persist
                                 with tunable lifetime; tau_bound = inf
                                 recovers immortal bound states]

Channel branching per event:
    photons (prob p_gamma): pair removed, 4-momentum -> escaped-photon
        ledger; active at ALL temperatures (late-time meson depletion).
    gluons (prob 1 - p_gamma): only while T_eff > T_chem; pair removed and a
        new matching-color q-qbar pair is injected carrying the removed
        four-momentum, flavor drawn from thermal weights at T_eff subject to
        the 2 m_f <= M_inv threshold.  Below T_chem the gluon channel is off
        (cold gluonic annihilation would be a hadronic decay, outside the
        model), so the effective hazard is scaled by the open-channel weight.

Every event removes/adds color-neutral, flavor-conserving multisets; the
global color vector and net flavor numbers are asserted unchanged by the
Simulation driver after every reaction pass.
"""

import math

import numpy as np
from numba import njit

from .potentials import pair_potential
from .units import FLAVOR_NAMES, FLAVOR_MASSES, bessel_k2


@njit(cache=True)
def find_candidate_pairs_csr(x, labels, flavors, alive, L, d_ann,
                             offsets, indices):
    """CSR version: iterate quark rows only; each (q, qbar) pair appears
    exactly once (from the quark side)."""
    n = x.shape[0]
    cap = 4 * n
    ii = np.empty(cap, dtype=np.int64)
    jj = np.empty(cap, dtype=np.int64)
    rr = np.empty(cap, dtype=np.float64)
    count = 0
    d2 = d_ann * d_ann
    for i in range(n):
        if not alive[i] or labels[i] >= 3:
            continue
        for k in range(offsets[i], offsets[i + 1]):
            j = indices[k]
            if not alive[j] or labels[j] < 3:
                continue
            if labels[j] != labels[i] + 3 or flavors[j] != flavors[i]:
                continue
            dx = x[i, 0] - x[j, 0]
            dy = x[i, 1] - x[j, 1]
            dz = x[i, 2] - x[j, 2]
            dx -= L * round(dx / L)
            dy -= L * round(dy / L)
            dz -= L * round(dz / L)
            r2 = dx * dx + dy * dy + dz * dz
            if r2 < d2 and count < cap:
                ii[count] = i
                jj[count] = j
                rr[count] = math.sqrt(r2)
                count += 1
    return ii[:count], jj[:count], rr[:count]


@njit(cache=True)
def find_candidate_pairs(x, labels, flavors, alive, L, d_ann):
    """Matching-color same-flavor q-qbar pairs with separation < d_ann.

    Returns (i_arr, j_arr, r_arr) with i a quark, j an antiquark.
    """
    n = x.shape[0]
    cap = 4 * n
    ii = np.empty(cap, dtype=np.int64)
    jj = np.empty(cap, dtype=np.int64)
    rr = np.empty(cap, dtype=np.float64)
    count = 0
    d2 = d_ann * d_ann
    for i in range(n):
        if not alive[i] or labels[i] >= 3:
            continue
        for j in range(n):
            if not alive[j] or labels[j] < 3:
                continue
            if labels[j] != labels[i] + 3 or flavors[j] != flavors[i]:
                continue
            dx = x[i, 0] - x[j, 0]
            dy = x[i, 1] - x[j, 1]
            dz = x[i, 2] - x[j, 2]
            dx -= L * round(dx / L)
            dy -= L * round(dy / L)
            dz -= L * round(dz / L)
            r2 = dx * dx + dy * dy + dz * dz
            if r2 < d2 and count < cap:
                ii[count] = i
                jj[count] = j
                rr[count] = math.sqrt(r2)
                count += 1
    return ii[:count], jj[:count], rr[:count]


def pair_energy(p, mass, i, j, v_ij):
    """E_pair = M_inv - m_i - m_j + V_ij (bound iff negative)."""
    e_i = math.sqrt(p[i] @ p[i] + mass[i] ** 2)
    e_j = math.sqrt(p[j] @ p[j] + mass[j] ** 2)
    ptot = p[i] + p[j]
    m_inv2 = (e_i + e_j) ** 2 - ptot @ ptot
    m_inv = math.sqrt(max(m_inv2, 0.0))
    return m_inv - mass[i] - mass[j] + v_ij, m_inv


def thermal_flavor_probs(t_eff, m_inv, flavors=("u", "d", "s", "c")):
    """Thermal weights n_f ~ m^2 T K2(m/T) restricted to 2 m_f <= M_inv."""
    weights = []
    names = []
    for f in flavors:
        m = FLAVOR_MASSES[f]
        if 2.0 * m <= m_inv:
            weights.append(m * m * t_eff * bessel_k2(m / t_eff))
            names.append(f)
    if not names:
        return [], []
    total = sum(weights)
    return names, [w / total for w in weights]


def make_injected_pair(m_new, p_total, e_total, d_ann, L, rng):
    """Kinematics for an injected q-qbar pair of mass m_new each, carrying
    total lab 4-momentum (e_total, p_total): back-to-back in the pair CM,
    boosted to the lab.  Returns (x1, x2, p1, p2) positions/momenta."""
    m_inv2 = e_total**2 - p_total @ p_total
    m_inv = math.sqrt(max(m_inv2, (2.0 * m_new) ** 2))
    p_star = math.sqrt(max(m_inv**2 / 4.0 - m_new**2, 0.0))
    e_star = math.sqrt(p_star**2 + m_new**2)

    # isotropic CM direction
    cos_t = rng.uniform(-1.0, 1.0)
    phi = rng.uniform(0.0, 2.0 * math.pi)
    sin_t = math.sqrt(1.0 - cos_t**2)
    n_hat = np.array([sin_t * math.cos(phi), sin_t * math.sin(phi), cos_t])
    p1_star = p_star * n_hat
    p2_star = -p1_star

    # boost by beta = p_total / e_total
    beta = p_total / e_total
    b2 = beta @ beta
    if b2 > 1e-14:
        gamma = 1.0 / math.sqrt(1.0 - b2)
        for pv in (p1_star, p2_star):
            bp = beta @ pv
            pv += ((gamma - 1.0) * bp / b2 + gamma * e_star) * beta
    p1, p2 = p1_star, p2_star

    x1 = rng.uniform(0.0, L, 3)
    # random orientation for the separation
    cos_t = rng.uniform(-1.0, 1.0)
    phi = rng.uniform(0.0, 2.0 * math.pi)
    sin_t = math.sqrt(1.0 - cos_t**2)
    sep = d_ann * np.array([sin_t * math.cos(phi), sin_t * math.sin(phi), cos_t])
    x2 = (x1 + sep) % L
    return x1, x2, p1, p2


class ReactionEngine:
    """Stateful reaction handler operating on the Simulation's state arrays."""

    def __init__(self, params, rng):
        self.lambda_scatt = params["lambda_scatt"]
        self.tau_bound = params["tau_bound"]  # may be math.inf
        self.p_gamma = params["p_gamma"]
        self.t_chem = params["t_chem"]
        self.d_ann = params["d_ann"]
        self.rng = rng
        self.photon_ledger = []   # (t, E, px, py, pz)
        self.gluon_events = []    # (t, flavor_removed, flavor_injected)
        self.n_annihilations = 0
        self.n_reinjections = 0

    def ledger_energy(self):
        return sum(rec[1] for rec in self.photon_ledger)

    def step(self, state, t, dt_react, t_eff, potential_pars, csr=None):
        """One reaction pass over candidate pairs.  Mutates state in place.

        state: dict with x, p, labels, flavors, mass, alive (numpy arrays).
        potential_pars: (c_table, alpha_s, lam_d, r0, r_cut, L)
        csr: optional (offsets, indices) neighbor list covering d_ann;
             when given, the O(N^2) candidate scan is skipped.
        Returns number of annihilation events this pass.
        """
        x, p = state["x"], state["p"]
        labels, flavors = state["labels"], state["flavors"]
        mass, alive = state["mass"], state["alive"]
        c_table, alpha_s, lam_d, r0, r_cut, L = potential_pars

        if csr is not None:
            ii, jj, rr = find_candidate_pairs_csr(
                x, labels, flavors, alive, L, self.d_ann, csr[0], csr[1]
            )
        else:
            ii, jj, rr = find_candidate_pairs(
                x, labels, flavors, alive, L, self.d_ann
            )
        if len(ii) == 0:
            return 0

        # channel availability: photons always; gluons only while hot
        gluons_open = t_eff > self.t_chem
        branch_open = self.p_gamma + (1.0 - self.p_gamma) * (1.0 if gluons_open else 0.0)
        if branch_open == 0.0:
            return 0

        order = self.rng.permutation(len(ii))
        removed = set()
        n_events = 0
        for idx in order:
            i, j = int(ii[idx]), int(jj[idx])
            if i in removed or j in removed:
                continue
            v_ij = pair_potential(
                rr[idx], c_table[labels[i], labels[j]], alpha_s, lam_d, r0, r_cut
            )
            e_pair, m_inv = pair_energy(p, mass, i, j, v_ij)
            if e_pair > 0.0:
                rate = self.lambda_scatt / 3.0
            elif math.isfinite(self.tau_bound):
                rate = 1.0 / (3.0 * self.tau_bound)
            else:
                rate = 0.0
            rate *= branch_open
            if rate <= 0.0:
                continue
            if self.rng.random() >= 1.0 - math.exp(-rate * dt_react):
                continue

            # --- event fires: pick channel among open ones ---
            p_gamma_eff = self.p_gamma / branch_open
            e_i = math.sqrt(p[i] @ p[i] + mass[i] ** 2)
            e_j = math.sqrt(p[j] @ p[j] + mass[j] ** 2)
            e_tot = e_i + e_j
            p_tot = p[i] + p[j]

            alive[i] = False
            alive[j] = False
            removed.update((i, j))
            self.n_annihilations += 1
            n_events += 1

            if self.rng.random() < p_gamma_eff:
                # photons escape
                self.photon_ledger.append(
                    (t, e_tot, p_tot[0], p_tot[1], p_tot[2])
                )
            else:
                # gluon channel: re-inject thermal-flavor pair
                # (invariant mass sets the 2 m_f threshold)
                names, probs = thermal_flavor_probs(t_eff, m_inv)
                if not names:
                    # below every threshold (cannot happen for same-mass pair,
                    # since m_inv >= 2m); ledger the energy defensively
                    self.photon_ledger.append(
                        (t, e_tot, p_tot[0], p_tot[1], p_tot[2])
                    )
                    continue
                flavor_new = self.rng.choice(names, p=probs)
                m_new = FLAVOR_MASSES[flavor_new]
                x1, x2, p1, p2 = make_injected_pair(
                    m_new, p_tot, e_tot, self.d_ann, L, self.rng
                )
                col = int(self.rng.integers(0, 3))
                self._inject(state, x1, p1, col, flavor_new)
                self._inject(state, x2, p2, col + 3, flavor_new)
                self.n_reinjections += 1
                self.gluon_events.append(
                    (t, FLAVOR_NAMES[int(flavors[i])], flavor_new)
                )
        return n_events

    def _inject(self, state, x_new, p_new, label, flavor_name):
        """Reuse a dead slot or append a new particle."""
        from .units import FLAVOR_IDS

        dead = np.where(~state["alive"])[0]
        m_new = FLAVOR_MASSES[flavor_name]
        if len(dead):
            k = int(dead[0])
            state["x"][k] = x_new
            state["p"][k] = p_new
            state["labels"][k] = label
            state["flavors"][k] = FLAVOR_IDS[flavor_name]
            state["mass"][k] = m_new
            state["alive"][k] = True
        else:
            state["x"] = np.vstack([state["x"], x_new])
            state["p"] = np.vstack([state["p"], p_new])
            state["labels"] = np.append(state["labels"], label)
            state["flavors"] = np.append(state["flavors"], FLAVOR_IDS[flavor_name])
            state["mass"] = np.append(state["mass"], m_new)
            state["alive"] = np.append(state["alive"], True)
