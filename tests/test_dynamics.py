import numpy as np
import pytest

from diquark_md import color
from diquark_md.integrator import (
    drift,
    effective_temperature,
    kick,
    kinetic_energy,
    sample_juttner_momentum,
    zero_total_momentum,
    expansion_substep,
)
from diquark_md.potentials import compute_forces, pair_potential, pair_dvdr

C_TABLE = color.build_c_table()
ALPHA = 0.4
LAM_D = 0.4
R0 = 0.2


def test_force_is_minus_grad_v():
    """pair_dvdr matches numerical derivative of the unshifted potential."""
    for r in [0.05, 0.2, 0.5, 1.0, 2.0]:
        h = 1e-6
        big_cut = 1e9  # unshifted
        v_p = pair_potential(r + h, -2.0 / 3.0, ALPHA, LAM_D, R0, big_cut)
        v_m = pair_potential(r - h, -2.0 / 3.0, ALPHA, LAM_D, R0, big_cut)
        num = (v_p - v_m) / (2 * h)
        ana = pair_dvdr(r, -2.0 / 3.0, ALPHA, LAM_D, R0)
        assert ana == pytest.approx(num, rel=1e-5)


def test_force_bounded_at_origin():
    """Plummer softening bounds |dV/dr| everywhere; at r=0 the screening
    exponential leaves the finite cusp slope A * (-1/(lam_d * r0))."""
    from diquark_md.units import HBARC

    a = 0.5 * (-2.0 / 3.0) * ALPHA * HBARC
    expected = a * (-1.0 / (LAM_D * R0))
    assert pair_dvdr(0.0, -2.0 / 3.0, ALPHA, LAM_D, R0) == pytest.approx(expected)
    # monotonically bounded near the origin
    rs = np.linspace(0.0, 0.5, 200)
    vals = np.array([pair_dvdr(r, -2.0 / 3.0, ALPHA, LAM_D, R0) for r in rs])
    assert np.all(np.abs(vals) < 10.0)


def _nve_system(n=64, seed=1):
    rng = np.random.default_rng(seed)
    L = 4.0
    x = rng.uniform(0, L, (n, 3))
    labels = np.repeat(np.arange(6), n // 6 + 1)[:n].astype(np.int64)
    mass = np.full(n, 0.34)
    p = sample_juttner_momentum(0.34, 0.2, n, rng)
    alive = np.ones(n, dtype=np.bool_)
    zero_total_momentum(p, alive)
    return x, p, labels, mass, alive, L


def _total_energy(x, p, labels, mass, alive, L, r_cut):
    _, pot = compute_forces(x, labels, alive, C_TABLE, L, ALPHA, LAM_D, R0, r_cut)
    return kinetic_energy(p, mass, alive) + pot


def test_nve_energy_and_momentum_conservation():
    x, p, labels, mass, alive, L = _nve_system()
    r_cut = min(L / 2, 6 * LAM_D)
    dt = 0.005
    e0 = _total_energy(x, p, labels, mass, alive, L, r_cut)
    p0 = p[alive].sum(axis=0)
    forces, _ = compute_forces(x, labels, alive, C_TABLE, L, ALPHA, LAM_D, R0, r_cut)
    for _ in range(2000):
        kick(p, forces, alive, 0.5 * dt)
        drift(x, p, mass, alive, dt, L)
        forces, _ = compute_forces(x, labels, alive, C_TABLE, L, ALPHA, LAM_D, R0, r_cut)
        kick(p, forces, alive, 0.5 * dt)
    e1 = _total_energy(x, p, labels, mass, alive, L, r_cut)
    p1 = p[alive].sum(axis=0)
    assert abs(e1 - e0) / abs(e0) < 1e-4
    assert np.allclose(p0, p1, atol=1e-10)


def test_juttner_sampler_temperature():
    rng = np.random.default_rng(2)
    m, T = 0.34, 0.2
    p = sample_juttner_momentum(m, T, 200_000, rng)
    mass = np.full(len(p), m)
    alive = np.ones(len(p), dtype=np.bool_)
    # T_eff = <p^2/3E> is exact for Juttner
    assert effective_temperature(p, mass, alive) == pytest.approx(T, rel=5e-3)


def test_free_expansion_redshift():
    """With forces off, momenta redshift exactly as 1/a."""
    rng = np.random.default_rng(3)
    n = 100
    L = 5.0
    x = rng.uniform(0, L, (n, 3))
    p = sample_juttner_momentum(0.34, 0.2, n, rng)
    alive = np.ones(n, dtype=np.bool_)
    p_init = p.copy()
    a_grid = 1.0 + 0.1 * np.arange(0, 51)
    lam = 0.4
    for a_old, a_new in zip(a_grid[:-1], a_grid[1:]):
        L, lam = expansion_substep(x, p, alive, L, 0.4, a_old, a_new)
    assert np.allclose(p, p_init / a_grid[-1], rtol=1e-12)
    assert lam == pytest.approx(0.4 * a_grid[-1])
    assert L == pytest.approx(5.0 * a_grid[-1])


def test_two_body_bound_orbit_energy():
    """A bound matched q-qbar pair conserves energy to high precision."""
    L = 100.0  # effectively isolated
    x = np.array([[50.0, 50.0, 50.0], [50.35, 50.0, 50.0]])
    labels = np.array([0, 3], dtype=np.int64)
    mass = np.array([0.34, 0.34])
    p = np.array([[0.0, 0.02, 0.0], [0.0, -0.02, 0.0]])
    alive = np.ones(2, dtype=np.bool_)
    r_cut = 6 * LAM_D
    dt = 0.001
    e0 = _total_energy(x, p, labels, mass, alive, L, r_cut)
    assert e0 - mass.sum() < 0  # actually bound
    forces, _ = compute_forces(x, labels, alive, C_TABLE, L, ALPHA, LAM_D, R0, r_cut)
    for _ in range(20000):
        kick(p, forces, alive, 0.5 * dt)
        drift(x, p, mass, alive, dt, L)
        forces, _ = compute_forces(x, labels, alive, C_TABLE, L, ALPHA, LAM_D, R0, r_cut)
        kick(p, forces, alive, 0.5 * dt)
    e1 = _total_energy(x, p, labels, mass, alive, L, r_cut)
    assert abs(e1 - e0) < 1e-6
