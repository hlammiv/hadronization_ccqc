"""Simulation driver: init -> equilibrate -> expand/react -> measure.

Phases
------
1. init: thermal u,d,s at T0 (Boltzmann weights), N_ccbar injected charm
   pairs, exact color balance (global singlet), Maxwell-Juttner momenta.
2. equilibrate: Andersen thermostat, no reactions, no expansion.
3. production: NVE leapfrog + Hubble expand-and-redshift + Poisson-hazard
   reactions; cluster measurement in the dilute late phase.

Invariants asserted every reaction pass: global color vector unchanged,
net flavor numbers zero.
"""

import math
from collections import Counter

import numpy as np

from . import color
from .clusters import PathwayTracker, PersistenceTracker, find_clusters
from .integrator import (
    andersen_thermostat,
    drift,
    effective_temperature,
    kick,
    kinetic_energy,
    sample_juttner_momentum,
    zero_total_momentum,
    expansion_substep,
)
from .neighbors import NeighborList
from .potentials import compute_forces_csr
from .reactions import ReactionEngine
from .units import FLAVOR_IDS, FLAVOR_MASSES, thermal_flavor_weights


class Simulation:
    def __init__(self, cfg, seed=0):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.c_table = color.build_c_table(cfg.get("color_mode", "mean_channel"))
        self._init_particles()
        self.a = 1.0
        self.t = 0.0
        self.lam_d = cfg["lam_d0"]
        self.reactions = ReactionEngine(
            dict(
                lambda_scatt=cfg["lambda_scatt"],
                tau_bound=cfg.get("tau_bound", math.inf),
                p_gamma=cfg["p_gamma"],
                t_chem=cfg["t_chem"],
                d_ann=cfg["d_ann"],
            ),
            self.rng,
        )
        self.persistence = PersistenceTracker(cfg.get("n_persist", 3))
        self.pathways = PathwayTracker(window=cfg.get("pathway_window", 1.0))
        self.nl = NeighborList(skin=cfg.get("neighbor_skin", 0.4))
        self.history = []          # per-snapshot dicts
        self.initial_color_vector = color.color_vector(
            self.state["labels"][self.state["alive"]]
        )

    # ------------------------------------------------------------------ init
    def _init_particles(self):
        cfg = self.cfg
        n_pairs = cfg["n_pairs"]              # light+strange q-qbar pairs
        n_ccbar = cfg.get("n_ccbar", 0)
        t0 = cfg["t0"]

        weights = thermal_flavor_weights(t0, ("u", "d", "s"))
        flavor_names = list(weights)
        counts = self.rng.multinomial(n_pairs, [weights[f] for f in flavor_names])

        q_flavors = []
        for f, c in zip(flavor_names, counts):
            q_flavors += [FLAVOR_IDS[f]] * int(c)
        q_flavors += [FLAVOR_IDS["c"]] * n_ccbar
        q_flavors = np.array(q_flavors, dtype=np.int64)
        n_each = len(q_flavors)

        # colors: exact balance within quarks and antiquarks
        q_colors = np.tile(np.arange(3), n_each // 3 + 1)[:n_each]
        self.rng.shuffle(q_colors)
        qbar_colors = np.tile(np.arange(3, 6), n_each // 3 + 1)[:n_each]
        self.rng.shuffle(qbar_colors)

        labels = np.concatenate([q_colors, qbar_colors]).astype(np.int64)
        flavors = np.concatenate([q_flavors, q_flavors]).astype(np.int64)
        mass = np.array(
            [FLAVOR_MASSES[list(FLAVOR_IDS)[f]] for f in flavors], dtype=float
        )

        n_total = 2 * n_each
        # box from total density n = n_total / L^3
        L = (n_total / cfg["density"]) ** (1.0 / 3.0)
        x = self.rng.uniform(0.0, L, (n_total, 3))
        p = np.empty((n_total, 3))
        for m in np.unique(mass):
            sel = mass == m
            p[sel] = sample_juttner_momentum(m, t0, int(sel.sum()), self.rng)
        alive = np.ones(n_total, dtype=np.bool_)
        zero_total_momentum(p, alive)

        self.state = dict(x=x, p=p, labels=labels, flavors=flavors,
                          mass=mass, alive=alive)
        self.L = L
        # exact global neutrality by construction
        assert color.is_neutral(labels)

    # ------------------------------------------------------------- utilities
    def _pot_pars(self):
        cfg = self.cfg
        r_cut = min(self.L / 2.0, 6.0 * self.lam_d)
        return (self.c_table, cfg["alpha_s"], self.lam_d, cfg["r0"], r_cut,
                self.L)

    def _csr(self):
        s = self.state
        r_cut = min(self.L / 2.0, 6.0 * self.lam_d)
        return self.nl.get(s["x"], s["alive"], self.L, r_cut)

    def _forces(self):
        c_table, alpha_s, lam_d, r0, r_cut, L = self._pot_pars()
        s = self.state
        offsets, indices = self._csr()
        return compute_forces_csr(s["x"], s["labels"], s["alive"], c_table,
                                  L, alpha_s, lam_d, r0, r_cut,
                                  offsets, indices)

    def t_eff(self):
        s = self.state
        return effective_temperature(s["p"], s["mass"], s["alive"])

    def _assert_invariants(self):
        s = self.state
        alive = s["alive"]
        cv = color.color_vector(s["labels"][alive])
        assert np.array_equal(cv, self.initial_color_vector), \
            f"color neutrality violated: {cv}"
        # net flavor of quarks minus antiquarks must vanish per flavor
        labs, flavs = s["labels"][alive], s["flavors"][alive]
        for f in np.unique(flavs):
            nq = int(np.sum((flavs == f) & (labs < 3)))
            nqbar = int(np.sum((flavs == f) & (labs >= 3)))
            assert nq == nqbar, f"net flavor {f}: {nq} vs {nqbar}"

    # ------------------------------------------------------------ main phases
    def equilibrate(self):
        cfg = self.cfg
        dt = cfg["dt"]
        n_steps = int(cfg["t_equil"] / dt)
        s = self.state
        forces, _ = self._forces()
        for step in range(n_steps):
            kick(s["p"], forces, s["alive"], 0.5 * dt)
            drift(s["x"], s["p"], s["mass"], s["alive"], dt, self.L)
            forces, _ = self._forces()
            kick(s["p"], forces, s["alive"], 0.5 * dt)
            andersen_thermostat(s["p"], s["mass"], s["alive"], cfg["t0"],
                                cfg.get("thermostat_nu", 1.0), dt, self.rng)
        zero_total_momentum(s["p"], s["alive"])

    def run_production(self, progress=None):
        cfg = self.cfg
        dt = cfg["dt"]
        h0 = cfg["h0"]
        a_max = cfg.get("a_max", 20.0)
        snap_every = int(round(cfg.get("snap_interval", 1.0) / dt))
        react_every = int(cfg.get("react_every", 1))
        measure_below_t = cfg.get("measure_below_t", cfg["t_chem"])

        s = self.state
        forces, _ = self._forces()
        step = 0
        while self.a < a_max:
            # leapfrog
            kick(s["p"], forces, s["alive"], 0.5 * dt)
            drift(s["x"], s["p"], s["mass"], s["alive"], dt, self.L)
            forces, _ = self._forces()
            kick(s["p"], forces, s["alive"], 0.5 * dt)

            # expansion substep
            a_new = 1.0 + h0 * (self.t + dt)
            self.L, self.lam_d = expansion_substep(
                s["x"], s["p"], s["alive"], self.L, cfg["lam_d0"], self.a, a_new
            )
            self.a = a_new
            self.t += dt
            step += 1

            # reactions
            if step % react_every == 0:
                n_ev = self.reactions.step(
                    s, self.t, react_every * dt, self.t_eff(),
                    self._pot_pars(), csr=self._csr()
                )
                if n_ev:
                    self.nl.invalidate()  # injections not in the pair list
                    self._assert_invariants()
                    forces, _ = self._forces()

            # snapshot + measurement
            if step % snap_every == 0:
                self._measure(measure_below_t)
                if progress:
                    progress(self)
        return self.finalize()

    def _measure(self, measure_below_t):
        cfg = self.cfg
        t_eff = self.t_eff()
        alive = self.state["alive"]
        flavs = self.state["flavors"][alive]
        rec = dict(t=self.t, a=self.a, t_eff=t_eff, L=self.L,
                   n_alive=int(alive.sum()),
                   n_s=int((flavs == 2).sum()),
                   n_c=int((flavs == 3).sum()),
                   n_annihilations=self.reactions.n_annihilations,
                   n_reinjections=self.reactions.n_reinjections,
                   n_photons=len(self.reactions.photon_ledger))
        if t_eff < measure_below_t:
            c_table, alpha_s, lam_d, r0, r_cut, L = self._pot_pars()
            s = self.state
            clusters, n_unres, edges = find_clusters(
                s["x"], s["p"], s["labels"], s["flavors"], s["mass"],
                s["alive"], c_table, L, alpha_s, lam_d, r0, r_cut,
                r_cl=cfg.get("r_cl_factor", 3.0) * lam_d,
                delta_e_th=cfg.get("delta_e_th", 0.0),
                return_edges=True, csr=self._csr(),
            )
            self.persistence.update(clusters)
            self.pathways.update(self.t, edges, clusters, s["labels"])
            rec["species"] = dict(Counter(c["species"] for c in clusters))
            rec["n_unresolved"] = n_unres
        self.history.append(rec)

    def finalize(self):
        counts, exotic_counts = self.persistence.tally()
        final_clusters = self.persistence.persistent_clusters()
        dq_frac, n_baryons = self.pathways.diquark_first_fraction(final_clusters)
        s = self.state
        alive = s["alive"]
        n_bound = sum(len(c["members"]) for c in final_clusters)
        return dict(
            species_counts=counts,
            exotic_counts=exotic_counts,
            clusters=final_clusters,
            diquark_first_fraction=dq_frac,
            n_pathway_baryons=n_baryons,
            n_alive_final=int(alive.sum()),
            n_free_final=int(alive.sum()) - n_bound,
            photon_ledger=list(self.reactions.photon_ledger),
            gluon_events=list(self.reactions.gluon_events),
            history=self.history,
            t_final=self.t,
            a_final=self.a,
        )
