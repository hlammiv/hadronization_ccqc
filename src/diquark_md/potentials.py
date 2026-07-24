"""Color-weighted, Plummer-softened, screened-Coulomb pair interaction.

    V_ij(r) = (C_ij / 2) alpha_s hbar-c exp(-r / lambda_D) / sqrt(r^2 + r0^2)
              - (same at r = r_cut)          [energy-shifted for continuity]

The shift makes V continuous at the cutoff so that pairs crossing r_cut do
not inject energy; forces are analytic everywhere and bounded at r -> 0 by
the Plummer softening r0.
"""

import math

import numpy as np
from numba import njit, prange

from .units import HBARC


@njit(cache=True)
def pair_potential(r, c_ij, alpha_s, lam_d, r0, r_cut):
    """Shifted pair potential; zero at and beyond r_cut."""
    if r >= r_cut:
        return 0.0
    a = 0.5 * c_ij * alpha_s * HBARC
    v = a * math.exp(-r / lam_d) / math.sqrt(r * r + r0 * r0)
    v_cut = a * math.exp(-r_cut / lam_d) / math.sqrt(r_cut * r_cut + r0 * r0)
    return v - v_cut


@njit(cache=True)
def pair_dvdr(r, c_ij, alpha_s, lam_d, r0):
    """dV/dr of the unshifted potential (the shift is constant)."""
    a = 0.5 * c_ij * alpha_s * HBARC
    s2 = r * r + r0 * r0
    s = math.sqrt(s2)
    return a * math.exp(-r / lam_d) * (-1.0 / (lam_d * s) - r / (s2 * s))


@njit(cache=True, parallel=True)
def compute_forces(x, labels, alive, c_table, L, alpha_s, lam_d, r0, r_cut):
    """Forces and total potential energy, O(N^2) with minimum image.

    Returns (forces[N,3], potential_energy).
    """
    n = x.shape[0]
    forces = np.zeros((n, 3))
    pot = np.zeros(n)
    r_cut2 = r_cut * r_cut
    for i in prange(n):
        if not alive[i]:
            continue
        fx = 0.0
        fy = 0.0
        fz = 0.0
        pe = 0.0
        li = labels[i]
        for j in range(n):
            if j == i or not alive[j]:
                continue
            dx = x[i, 0] - x[j, 0]
            dy = x[i, 1] - x[j, 1]
            dz = x[i, 2] - x[j, 2]
            dx -= L * round(dx / L)
            dy -= L * round(dy / L)
            dz -= L * round(dz / L)
            r2 = dx * dx + dy * dy + dz * dz
            if r2 >= r_cut2 or r2 == 0.0:
                continue
            r = math.sqrt(r2)
            c_ij = c_table[li, labels[j]]
            dvdr = pair_dvdr(r, c_ij, alpha_s, lam_d, r0)
            f = -dvdr / r
            fx += f * dx
            fy += f * dy
            fz += f * dz
            pe += 0.5 * pair_potential(r, c_ij, alpha_s, lam_d, r0, r_cut)
        forces[i, 0] = fx
        forces[i, 1] = fy
        forces[i, 2] = fz
        pot[i] = pe
    return forces, pot.sum()


@njit(cache=True, parallel=True)
def compute_forces_csr(x, labels, alive, c_table, L, alpha_s, lam_d, r0,
                       r_cut, offsets, indices):
    """Forces and total potential energy over a CSR neighbor list.

    Identical output to compute_forces when the list covers r_cut.
    """
    n = x.shape[0]
    forces = np.zeros((n, 3))
    pot = np.zeros(n)
    r_cut2 = r_cut * r_cut
    for i in prange(n):
        if not alive[i]:
            continue
        fx = 0.0
        fy = 0.0
        fz = 0.0
        pe = 0.0
        li = labels[i]
        for k in range(offsets[i], offsets[i + 1]):
            j = indices[k]
            if not alive[j]:
                continue
            dx = x[i, 0] - x[j, 0]
            dy = x[i, 1] - x[j, 1]
            dz = x[i, 2] - x[j, 2]
            dx -= L * round(dx / L)
            dy -= L * round(dy / L)
            dz -= L * round(dz / L)
            r2 = dx * dx + dy * dy + dz * dz
            if r2 >= r_cut2 or r2 == 0.0:
                continue
            r = math.sqrt(r2)
            c_ij = c_table[li, labels[j]]
            dvdr = pair_dvdr(r, c_ij, alpha_s, lam_d, r0)
            f = -dvdr / r
            fx += f * dx
            fy += f * dy
            fz += f * dz
            pe += 0.5 * pair_potential(r, c_ij, alpha_s, lam_d, r0, r_cut)
        forces[i, 0] = fx
        forces[i, 1] = fy
        forces[i, 2] = fz
        pot[i] = pe
    return forces, pot.sum()


@njit(cache=True)
def pair_separation(x, i, j, L):
    """Minimum-image distance between particles i and j."""
    dx = x[i, 0] - x[j, 0]
    dy = x[i, 1] - x[j, 1]
    dz = x[i, 2] - x[j, 2]
    dx -= L * round(dx / L)
    dy -= L * round(dy / L)
    dz -= L * round(dz / L)
    return math.sqrt(dx * dx + dy * dy + dz * dz)
