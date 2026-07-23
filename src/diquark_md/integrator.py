"""Relativistic leapfrog, Maxwell-Juttner sampling, Andersen-style thermostat,
and the Hubble expand-and-redshift substep.

The Hamiltonian H = sum_i sqrt(p_i^2 + m_i^2) + V({x}) is separable, so
kick-drift-kick leapfrog with xdot = p/E is symplectic and time-reversible.

Equilibration uses an Andersen-type thermostat (stochastic re-draws of
individual momenta from the Maxwell-Juttner distribution at rate nu): it is
ergodic and produces the exact Juttner marginal by construction.  Production
phases run NVE (+ the expansion map); all dissipation is physics we control.
"""

import numpy as np
from numba import njit, prange


@njit(cache=True, parallel=True)
def kick(p, forces, alive, dt):
    n = p.shape[0]
    for i in prange(n):
        if alive[i]:
            p[i, 0] += forces[i, 0] * dt
            p[i, 1] += forces[i, 1] * dt
            p[i, 2] += forces[i, 2] * dt


@njit(cache=True, parallel=True)
def drift(x, p, mass, alive, dt, L):
    n = x.shape[0]
    for i in prange(n):
        if alive[i]:
            e = np.sqrt(p[i, 0] ** 2 + p[i, 1] ** 2 + p[i, 2] ** 2 + mass[i] ** 2)
            for k in range(3):
                x[i, k] += p[i, k] / e * dt
                # wrap into [0, L)
                x[i, k] -= L * np.floor(x[i, k] / L)


@njit(cache=True)
def kinetic_energy(p, mass, alive):
    """Total relativistic kinetic energy sum_i (E_i - m_i)."""
    total = 0.0
    n = p.shape[0]
    for i in range(n):
        if alive[i]:
            e = np.sqrt(p[i, 0] ** 2 + p[i, 1] ** 2 + p[i, 2] ** 2 + mass[i] ** 2)
            total += e - mass[i]
    return total


@njit(cache=True)
def effective_temperature(p, mass, alive):
    """Relativistic virial estimator T_eff = <p^2 / (3 E)>, exact for Juttner."""
    total = 0.0
    count = 0
    n = p.shape[0]
    for i in range(n):
        if alive[i]:
            p2 = p[i, 0] ** 2 + p[i, 1] ** 2 + p[i, 2] ** 2
            e = np.sqrt(p2 + mass[i] ** 2)
            total += p2 / (3.0 * e)
            count += 1
    return total / count if count > 0 else 0.0


def sample_juttner_momentum(mass, temperature, size, rng):
    """Sample |p| from f(p) ~ p^2 exp(-(E - m)/T) by inverse-CDF on a grid,
    then isotropic directions.  Returns (size, 3) momenta."""
    p_max = 15.0 * temperature + 8.0 * np.sqrt(mass * temperature) + 3.0 * temperature
    grid = np.linspace(0.0, p_max, 4096)
    e = np.sqrt(grid**2 + mass**2)
    pdf = grid**2 * np.exp(-(e - mass) / temperature)
    cdf = np.cumsum(pdf)
    cdf /= cdf[-1]
    u = rng.random(size)
    pmag = np.interp(u, cdf, grid)
    cos_t = rng.uniform(-1.0, 1.0, size)
    phi = rng.uniform(0.0, 2.0 * np.pi, size)
    sin_t = np.sqrt(1.0 - cos_t**2)
    p = np.empty((size, 3))
    p[:, 0] = pmag * sin_t * np.cos(phi)
    p[:, 1] = pmag * sin_t * np.sin(phi)
    p[:, 2] = pmag * cos_t
    return p


def andersen_thermostat(p, mass, alive, temperature, nu, dt, rng):
    """Re-draw momenta of randomly selected particles from Juttner at T.

    Each alive particle is refreshed with probability nu*dt this step.
    """
    n = p.shape[0]
    hits = np.where(alive & (rng.random(n) < nu * dt))[0]
    for m in np.unique(mass[hits]):
        idx = hits[mass[hits] == m]
        p[idx] = sample_juttner_momentum(m, temperature, len(idx), rng)
    return len(hits)


def zero_total_momentum(p, alive):
    n_alive = int(np.sum(alive))
    if n_alive:
        p[alive] -= p[alive].sum(axis=0) / n_alive


def expansion_substep(x, p, alive, L, lam_d0, a_old, a_new):
    """Operator-split Hubble step: dilate positions and box, redshift peculiar
    momenta as 1/a, and grow the (comoving-fixed) screening length.

    Returns (L_new, lam_d_new).
    """
    ratio = a_new / a_old
    x[alive] *= ratio
    p[alive] /= ratio
    return L * ratio, lam_d0 * a_new
