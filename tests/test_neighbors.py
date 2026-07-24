"""Cell-list/Verlet CSR neighbor machinery: exactness vs brute force."""

import numpy as np
import pytest

from diquark_md import color
from diquark_md.neighbors import NeighborList, build_neighbor_csr
from diquark_md.potentials import compute_forces, compute_forces_csr
from diquark_md.reactions import find_candidate_pairs, find_candidate_pairs_csr
from diquark_md.clusters import _find_edges

C_TABLE = color.build_c_table()
ALPHA, LAM_D, R0 = 0.4, 0.4, 0.2


def _config(n, L, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, L, (n, 3))
    labels = np.tile(np.arange(6), n // 6 + 1)[:n].astype(np.int64)
    rng.shuffle(labels)
    flavors = rng.integers(0, 3, n).astype(np.int64)
    mass = np.full(n, 0.34)
    p = rng.normal(0, 0.3, (n, 3))
    alive = np.ones(n, dtype=np.bool_)
    alive[rng.integers(0, n, n // 20)] = False  # some dead slots
    return x, p, labels, flavors, mass, alive


@pytest.mark.parametrize("L", [10.0, 5.0])  # cell path and brute fallback
def test_forces_csr_exact(L):
    n = 400
    x, p, labels, flavors, mass, alive = _config(n, L)
    r_cut = 2.4
    offsets, indices = build_neighbor_csr(x, alive, L, r_cut + 0.4)
    f0, pe0 = compute_forces(x, labels, alive, C_TABLE, L, ALPHA, LAM_D,
                             R0, r_cut)
    f1, pe1 = compute_forces_csr(x, labels, alive, C_TABLE, L, ALPHA, LAM_D,
                                 R0, r_cut, offsets, indices)
    assert np.allclose(f0, f1, atol=1e-12, rtol=0)
    assert pe0 == pytest.approx(pe1, abs=1e-12)


def test_candidate_pairs_csr_exact():
    n = 400
    L = 6.0
    x, p, labels, flavors, mass, alive = _config(n, L, seed=3)
    d_ann = 0.5  # generous so pairs exist
    offsets, indices = build_neighbor_csr(x, alive, L, 2.8)
    i0, j0, r0_ = find_candidate_pairs(x, labels, flavors, alive, L, d_ann)
    i1, j1, r1_ = find_candidate_pairs_csr(x, labels, flavors, alive, L,
                                           d_ann, offsets, indices)
    assert set(zip(i0.tolist(), j0.tolist())) == set(zip(i1.tolist(), j1.tolist()))
    assert len(i0) > 0


def test_edges_csr_exact():
    n = 300
    L = 8.0
    x, p, labels, flavors, mass, alive = _config(n, L, seed=5)
    p *= 0.1  # cold enough for bound pairs
    r_cut, r_cl = 2.4, 1.2
    csr = build_neighbor_csr(x, alive, L, r_cut + 0.4)
    e0 = _find_edges(x, p, labels, mass, alive, C_TABLE, L, ALPHA, LAM_D,
                     R0, r_cut, r_cl)
    e1 = _find_edges(x, p, labels, mass, alive, C_TABLE, L, ALPHA, LAM_D,
                     R0, r_cut, r_cl, csr=csr)
    assert e0 == e1
    assert len(e0) > 0


def test_neighborlist_skin_rebuilds():
    """Drifting particles: nl.get() must always cover r_cut correctly, and
    the skin must avoid rebuilding every call."""
    n = 300
    L = 10.0
    rng = np.random.default_rng(7)
    x, p, labels, flavors, mass, alive = _config(n, L, seed=9)
    nl = NeighborList(skin=0.4)
    r_cut = 2.4
    n_calls = 40
    for step in range(n_calls):
        x += rng.normal(0, 0.02, x.shape)  # ~0.02 fm/step drift
        x %= L
        offsets, indices = nl.get(x, alive, L, r_cut)
        f0, pe0 = compute_forces(x, labels, alive, C_TABLE, L, ALPHA,
                                 LAM_D, R0, r_cut)
        f1, pe1 = compute_forces_csr(x, labels, alive, C_TABLE, L, ALPHA,
                                     LAM_D, R0, r_cut, offsets, indices)
        assert np.allclose(f0, f1, atol=1e-12, rtol=0)
    assert nl.n_builds < n_calls / 2  # the skin is doing its job


def test_neighborlist_hubble_rescale_free():
    """Pure box rescaling must not trigger rebuilds (covered radius
    scales with the box)."""
    n = 200
    L = 10.0
    x, p, labels, flavors, mass, alive = _config(n, L, seed=11)
    nl = NeighborList(skin=0.4)
    lam = LAM_D
    nl.get(x, alive, L, 6 * lam)
    builds0 = nl.n_builds
    for _ in range(20):
        x *= 1.01
        L *= 1.01
        lam *= 1.01
        offsets, indices = nl.get(x, alive, L, min(L / 2, 6 * lam))
        f0, _ = compute_forces(x, labels, alive, C_TABLE, L, ALPHA, lam,
                               R0, min(L / 2, 6 * lam))
        f1, _ = compute_forces_csr(x, labels, alive, C_TABLE, L, ALPHA, lam,
                                   R0, min(L / 2, 6 * lam), offsets, indices)
        assert np.allclose(f0, f1, atol=1e-12, rtol=0)
    assert nl.n_builds == builds0


def test_simulation_energy_conservation_with_neighborlist():
    """Leapfrog through the Simulation's CSR force path conserves energy."""
    from diquark_md.integrator import drift, kick, kinetic_energy
    from diquark_md.io import load_config
    from diquark_md.sim import Simulation

    cfg = load_config("configs/default.toml")
    cfg["n_pairs"] = 96
    cfg["t_equil"] = 2.0
    sim = Simulation(cfg, seed=1)
    sim.equilibrate()
    s = sim.state
    dt = cfg["dt"]

    def total_e():
        _, pe = sim._forces()
        return kinetic_energy(s["p"], s["mass"], s["alive"]) + pe

    e0 = total_e()
    forces, _ = sim._forces()
    for _ in range(1000):
        kick(s["p"], forces, s["alive"], 0.5 * dt)
        drift(s["x"], s["p"], s["mass"], s["alive"], dt, sim.L)
        forces, _ = sim._forces()
        kick(s["p"], forces, s["alive"], 0.5 * dt)
    e1 = total_e()
    assert abs(e1 - e0) / abs(e0) < 1e-4
    assert sim.nl.n_builds > 1  # rebuilds actually happened
