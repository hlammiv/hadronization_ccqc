"""Linked-cell + Verlet-skin neighbor lists (CSR layout).

The pair list covers all pairs within r_list = r_cut + skin.  It is
rebuilt only when (i) a particle's peculiar displacement since the last
build exceeds skin/2 (in box-scaled units, so pure Hubble rescaling never
triggers a rebuild -- the covered radius scales with the box), or (ii)
particles were injected (Simulation calls invalidate()).

CSR layout: for particle i, its neighbors are indices[offsets[i]:
offsets[i+1]].  Every pair appears in both rows, so force loops can
prange over i with no write contention; pair-unique iteration filters
j > i (or quark/antiquark roles).

For boxes smaller than 3 cells per side the cell decomposition is
degenerate and an O(N^2) brute builder is used instead -- same output,
same downstream kernels.
"""

import numpy as np
from numba import njit, prange


@njit(cache=True)
def _build_csr_brute(x, alive, L, r_list):
    n = x.shape[0]
    r2max = r_list * r_list
    counts = np.zeros(n + 1, dtype=np.int64)
    for i in range(n):
        if not alive[i]:
            continue
        c = 0
        for j in range(n):
            if j == i or not alive[j]:
                continue
            dx = x[i, 0] - x[j, 0]
            dy = x[i, 1] - x[j, 1]
            dz = x[i, 2] - x[j, 2]
            dx -= L * round(dx / L)
            dy -= L * round(dy / L)
            dz -= L * round(dz / L)
            if dx * dx + dy * dy + dz * dz < r2max:
                c += 1
        counts[i + 1] = c
    for i in range(n):
        counts[i + 1] += counts[i]
    indices = np.empty(counts[n], dtype=np.int64)
    for i in range(n):
        if not alive[i]:
            continue
        k = counts[i]
        for j in range(n):
            if j == i or not alive[j]:
                continue
            dx = x[i, 0] - x[j, 0]
            dy = x[i, 1] - x[j, 1]
            dz = x[i, 2] - x[j, 2]
            dx -= L * round(dx / L)
            dy -= L * round(dy / L)
            dz -= L * round(dz / L)
            if dx * dx + dy * dy + dz * dz < r2max:
                indices[k] = j
                k += 1
    return counts, indices


@njit(cache=True)
def _cell_of(xi, cell_size, ncell):
    c = int(xi / cell_size)
    if c >= ncell:
        c = ncell - 1
    if c < 0:
        c = 0
    return c


@njit(cache=True, parallel=True)
def _build_csr_cells(x, alive, L, r_list, ncell):
    n = x.shape[0]
    cell_size = L / ncell
    ncells3 = ncell * ncell * ncell
    r2max = r_list * r_list

    # bucket particles by cell (counting sort)
    cid = np.full(n, -1, dtype=np.int64)
    cell_count = np.zeros(ncells3 + 1, dtype=np.int64)
    for i in range(n):
        if alive[i]:
            cx = _cell_of(x[i, 0], cell_size, ncell)
            cy = _cell_of(x[i, 1], cell_size, ncell)
            cz = _cell_of(x[i, 2], cell_size, ncell)
            c = (cx * ncell + cy) * ncell + cz
            cid[i] = c
            cell_count[c + 1] += 1
    for c in range(ncells3):
        cell_count[c + 1] += cell_count[c]
    bucket = np.empty(n, dtype=np.int64)
    fill = cell_count[:-1].copy()
    for i in range(n):
        if cid[i] >= 0:
            bucket[fill[cid[i]]] = i
            fill[cid[i]] += 1

    # pass 1: neighbor counts
    counts = np.zeros(n + 1, dtype=np.int64)
    for i in prange(n):
        if not alive[i]:
            continue
        cx = _cell_of(x[i, 0], cell_size, ncell)
        cy = _cell_of(x[i, 1], cell_size, ncell)
        cz = _cell_of(x[i, 2], cell_size, ncell)
        c_i = 0
        for ox in range(-1, 2):
            for oy in range(-1, 2):
                for oz in range(-1, 2):
                    cc = ((((cx + ox) % ncell) * ncell + (cy + oy) % ncell)
                          * ncell + (cz + oz) % ncell)
                    for k in range(cell_count[cc], cell_count[cc + 1]):
                        j = bucket[k]
                        if j == i:
                            continue
                        dx = x[i, 0] - x[j, 0]
                        dy = x[i, 1] - x[j, 1]
                        dz = x[i, 2] - x[j, 2]
                        dx -= L * round(dx / L)
                        dy -= L * round(dy / L)
                        dz -= L * round(dz / L)
                        if dx * dx + dy * dy + dz * dz < r2max:
                            c_i += 1
        counts[i + 1] = c_i
    for i in range(n):
        counts[i + 1] += counts[i]

    # pass 2: fill
    indices = np.empty(counts[n], dtype=np.int64)
    for i in prange(n):
        if not alive[i]:
            continue
        cx = _cell_of(x[i, 0], cell_size, ncell)
        cy = _cell_of(x[i, 1], cell_size, ncell)
        cz = _cell_of(x[i, 2], cell_size, ncell)
        k_out = counts[i]
        for ox in range(-1, 2):
            for oy in range(-1, 2):
                for oz in range(-1, 2):
                    cc = ((((cx + ox) % ncell) * ncell + (cy + oy) % ncell)
                          * ncell + (cz + oz) % ncell)
                    for k in range(cell_count[cc], cell_count[cc + 1]):
                        j = bucket[k]
                        if j == i:
                            continue
                        dx = x[i, 0] - x[j, 0]
                        dy = x[i, 1] - x[j, 1]
                        dz = x[i, 2] - x[j, 2]
                        dx -= L * round(dx / L)
                        dy -= L * round(dy / L)
                        dz -= L * round(dz / L)
                        if dx * dx + dy * dy + dz * dz < r2max:
                            indices[k_out] = j
                            k_out += 1
    return counts, indices


def build_neighbor_csr(x, alive, L, r_list):
    """(offsets, indices) covering all alive pairs with r < r_list."""
    ncell = int(L / r_list)
    if ncell >= 3:
        return _build_csr_cells(x, alive, L, r_list, ncell)
    return _build_csr_brute(x, alive, L, r_list)


class NeighborList:
    """Verlet-skin manager around build_neighbor_csr.

    get() returns (offsets, indices) valid for interactions out to the
    caller's current r_cut; rebuilds lazily.
    """

    def __init__(self, skin=0.4):
        self.skin = skin
        self._offsets = None
        self._indices = None
        self._x = None
        self._L = None
        self._r_list = None
        self.n_builds = 0

    def invalidate(self):
        self._offsets = None

    def _needs_rebuild(self, x, alive, L, r_cut):
        if self._offsets is None or self._x.shape[0] != x.shape[0]:
            return True
        scale = L / self._L
        # covered radius scales with the box (pure Hubble rescale is free)
        if r_cut + 0.25 * self.skin * scale > self._r_list * scale:
            return True
        d = np.abs(x[alive] - self._x[alive] * scale)
        d = np.minimum(d, L - d)  # minimum image
        return float(d.max()) > 0.5 * self.skin * scale if len(d) else False

    def get(self, x, alive, L, r_cut):
        if self._needs_rebuild(x, alive, L, r_cut):
            self._r_list = r_cut + self.skin
            self._offsets, self._indices = build_neighbor_csr(
                x, alive, L, self._r_list
            )
            self._x = x.copy()
            self._L = L
            self.n_builds += 1
        return self._offsets, self._indices
