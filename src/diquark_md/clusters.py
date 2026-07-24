"""Color-singlet cluster identification (the measurement).

Pipeline per snapshot (run in the dilute late phase):

 1. candidate edges: pairs with r < r_cl that are channel-attractive
    (C_ij < 0) AND pairwise Hill-bound (M_inv - m_i - m_j + V_ij < 0);
 2. union-find -> pre-clusters;
 3. exact minimum-energy partition refinement within each pre-cluster into
    color-neutral parts or free singletons (ECRA-style, Dorso-Randrup
    PLB 301, 328): part energy = (M_inv,part - sum m) + sum_internal V;
    a part counts as a bound cluster iff E_part < -delta_e_th;
 4. species tally with flavor content and manifest-exotic classification.

Tetraquark vs two-mesons disambiguation IS the partition search: a (2,2)
part survives only if its 4-body energy beats every meson+meson (and
meson+2 free, ...) split of the same four particles.
"""

import math
from collections import Counter

import numpy as np
from numba import njit

from . import color
from .potentials import pair_potential, pair_separation
from .units import FLAVOR_NAMES

MAX_EXACT_SIZE = 12


@njit(cache=True)
def _edge_kernel(x, p, labels, mass, alive, c_table, L, alpha_s, lam_d, r0,
                 r_cut, r_cl):
    n = x.shape[0]
    cap = 32 * n
    out_i = np.empty(cap, dtype=np.int64)
    out_j = np.empty(cap, dtype=np.int64)
    count = 0
    r_cl2 = r_cl * r_cl
    for i in range(n):
        if not alive[i]:
            continue
        for j in range(i + 1, n):
            if not alive[j]:
                continue
            c_ij = c_table[labels[i], labels[j]]
            if c_ij >= 0.0:
                continue
            dx = x[i, 0] - x[j, 0]
            dy = x[i, 1] - x[j, 1]
            dz = x[i, 2] - x[j, 2]
            dx -= L * round(dx / L)
            dy -= L * round(dy / L)
            dz -= L * round(dz / L)
            r2 = dx * dx + dy * dy + dz * dz
            if r2 >= r_cl2:
                continue
            r = math.sqrt(r2)
            v = pair_potential(r, c_ij, alpha_s, lam_d, r0, r_cut)
            e_i = math.sqrt(p[i, 0] ** 2 + p[i, 1] ** 2 + p[i, 2] ** 2
                            + mass[i] ** 2)
            e_j = math.sqrt(p[j, 0] ** 2 + p[j, 1] ** 2 + p[j, 2] ** 2
                            + mass[j] ** 2)
            px = p[i, 0] + p[j, 0]
            py = p[i, 1] + p[j, 1]
            pz = p[i, 2] + p[j, 2]
            m_inv2 = (e_i + e_j) ** 2 - (px * px + py * py + pz * pz)
            m_inv = math.sqrt(max(m_inv2, 0.0))
            if m_inv - mass[i] - mass[j] + v < 0.0 and count < cap:
                out_i[count] = i
                out_j[count] = j
                count += 1
    return out_i[:count], out_j[:count]


@njit(cache=True)
def _edge_kernel_csr(x, p, labels, mass, alive, c_table, L, alpha_s, lam_d,
                     r0, r_cut, r_cl, offsets, indices):
    n = x.shape[0]
    cap = 32 * n
    out_i = np.empty(cap, dtype=np.int64)
    out_j = np.empty(cap, dtype=np.int64)
    count = 0
    r_cl2 = r_cl * r_cl
    for i in range(n):
        if not alive[i]:
            continue
        for k in range(offsets[i], offsets[i + 1]):
            j = indices[k]
            if j <= i or not alive[j]:
                continue
            c_ij = c_table[labels[i], labels[j]]
            if c_ij >= 0.0:
                continue
            dx = x[i, 0] - x[j, 0]
            dy = x[i, 1] - x[j, 1]
            dz = x[i, 2] - x[j, 2]
            dx -= L * round(dx / L)
            dy -= L * round(dy / L)
            dz -= L * round(dz / L)
            r2 = dx * dx + dy * dy + dz * dz
            if r2 >= r_cl2:
                continue
            r = math.sqrt(r2)
            v = pair_potential(r, c_ij, alpha_s, lam_d, r0, r_cut)
            e_i = math.sqrt(p[i, 0] ** 2 + p[i, 1] ** 2 + p[i, 2] ** 2
                            + mass[i] ** 2)
            e_j = math.sqrt(p[j, 0] ** 2 + p[j, 1] ** 2 + p[j, 2] ** 2
                            + mass[j] ** 2)
            px = p[i, 0] + p[j, 0]
            py = p[i, 1] + p[j, 1]
            pz = p[i, 2] + p[j, 2]
            m_inv2 = (e_i + e_j) ** 2 - (px * px + py * py + pz * pz)
            m_inv = math.sqrt(max(m_inv2, 0.0))
            if m_inv - mass[i] - mass[j] + v < 0.0 and count < cap:
                out_i[count] = i
                out_j[count] = j
                count += 1
    return out_i[:count], out_j[:count]


def _find_edges(x, p, labels, mass, alive, c_table, L, alpha_s, lam_d, r0,
                r_cut, r_cl, csr=None):
    """Hill-bound attractive edges among alive particles."""
    if csr is not None:
        ei, ej = _edge_kernel_csr(x, p, labels, mass, alive, c_table, L,
                                  alpha_s, lam_d, r0, r_cut, r_cl,
                                  csr[0], csr[1])
    else:
        ei, ej = _edge_kernel(x, p, labels, mass, alive, c_table, L, alpha_s,
                              lam_d, r0, r_cut, r_cl)
    return sorted(zip(ei.tolist(), ej.tolist()))


class _UnionFind:
    def __init__(self, items):
        self.parent = {i: i for i in items}

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i, j):
        ri, rj = self.find(i), self.find(j)
        if ri != rj:
            self.parent[ri] = rj


def _part_energy(members, x, p, labels, mass, c_table, L, alpha_s, lam_d,
                 r0, r_cut):
    """E_part = internal CM kinetic energy + internal potential energy.

    Free singleton convention: E = 0 (energies relative to dispersed-at-rest).
    """
    if len(members) == 1:
        return 0.0
    e_tot = 0.0
    ptot = np.zeros(3)
    msum = 0.0
    for i in members:
        e_tot += math.sqrt(p[i] @ p[i] + mass[i] ** 2)
        ptot += p[i]
        msum += mass[i]
    m_inv = math.sqrt(max(e_tot**2 - ptot @ ptot, 0.0))
    ke_cm = m_inv - msum
    v_int = 0.0
    for a in range(len(members)):
        for b in range(a + 1, len(members)):
            i, j = members[a], members[b]
            r = pair_separation(x, i, j, L)
            v_int += pair_potential(
                r, c_table[labels[i], labels[j]], alpha_s, lam_d, r0, r_cut
            )
    return ke_cm + v_int


def _refine_component(comp, x, p, labels, mass, c_table, L, alpha_s, lam_d,
                      r0, r_cut):
    """Minimum-total-energy partition of one pre-cluster into neutral parts
    or free singletons.  Returns list of (members_tuple, E_part) for parts
    of size >= 2 in the optimal partition."""
    comp = list(comp)
    comp_labels = [labels[i] for i in comp]
    energy_cache = {}

    def block_energy(block):
        if block not in energy_cache:
            members = [comp[k] for k in block]
            energy_cache[block] = _part_energy(
                members, x, p, labels, mass, c_table, L, alpha_s, lam_d,
                r0, r_cut,
            )
        return energy_cache[block]

    best = {"energy": math.inf, "partition": None}
    for partition in color.neutral_partitions(comp_labels):
        e = sum(block_energy(tuple(sorted(b))) for b in partition if len(b) > 1)
        if e < best["energy"]:
            best["energy"] = e
            best["partition"] = partition

    out = []
    for block in best["partition"]:
        if len(block) > 1:
            members = tuple(sorted(comp[k] for k in block))
            out.append((members, block_energy(tuple(sorted(block)))))
    return out


def find_clusters(x, p, labels, flavors, mass, alive, c_table, L, alpha_s,
                  lam_d, r0, r_cut, r_cl, delta_e_th=0.0, return_edges=False,
                  csr=None):
    """Full pipeline.  Returns (clusters, n_unresolved) where clusters is a
    list of dicts {members, energy, species, flavor_content, exotic};
    with return_edges=True also returns the Hill-bound attractive edge list
    (for the PathwayTracker).  csr: optional neighbor list covering r_cl."""
    edges = _find_edges(x, p, labels, mass, alive, c_table, L, alpha_s,
                        lam_d, r0, r_cut, r_cl, csr=csr)
    if not edges:
        return ([], 0, []) if return_edges else ([], 0)
    nodes = sorted({i for e in edges for i in e})
    uf = _UnionFind(nodes)
    for i, j in edges:
        uf.union(i, j)
    comps = {}
    for i in nodes:
        comps.setdefault(uf.find(i), []).append(i)

    clusters = []
    n_unresolved = 0
    for comp in comps.values():
        if len(comp) > MAX_EXACT_SIZE:
            n_unresolved += 1  # percolation guard: skip, report
            continue
        for members, e_part in _refine_component(
            comp, x, p, labels, mass, c_table, L, alpha_s, lam_d, r0, r_cut
        ):
            if e_part < -delta_e_th:
                clusters.append(_describe(members, e_part, labels, flavors))
    if return_edges:
        return clusters, n_unresolved, edges
    return clusters, n_unresolved


def _describe(members, e_part, labels, flavors):
    labs = [int(labels[i]) for i in members]
    nq = sum(1 for l in labs if l < 3)
    nqbar = len(labs) - nq
    # net flavor vector (quark minus antiquark, per flavor)
    net = Counter()
    content = []
    for i in members:
        f = FLAVOR_NAMES[int(flavors[i])]
        if labels[i] < 3:
            net[f] += 1
            content.append(f)
        else:
            net[f] -= 1
            content.append(f + "bar")
    return {
        "members": members,
        "energy": e_part,
        "species": color.species_of(labs),
        "flavor_content": tuple(sorted(content)),
        "exotic": _is_manifest_exotic(nq, nqbar, net),
    }


def _is_manifest_exotic(nq, nqbar, net):
    """A cluster is manifestly flavor-exotic if its net flavor vector cannot
    be carried by the minimal standard hadron with the same baryon number
    (B=0 -> q qbar; |B|=1 -> qqq / qbarqbarqbar).  E.g. cc-ubar-dbar
    (net c=+2) is exotic; c-cbar-u-ubar (net 0) is hidden-flavor."""
    b3 = nq - nqbar  # 3 x baryon number
    q_min = sum(v for v in net.values() if v > 0)
    qbar_min = sum(-v for v in net.values() if v < 0)
    if b3 == 0:
        return q_min > 1 or qbar_min > 1
    if b3 == 3:
        return qbar_min > 0 or q_min > 3
    if b3 == -3:
        return q_min > 0 or qbar_min > 3
    # higher-B clusters (dibaryons etc.): compare against B nucleon-like sets
    if b3 % 3 == 0 and b3 > 0:
        return qbar_min > 0
    if b3 % 3 == 0 and b3 < 0:
        return q_min > 0
    return False


class PathwayTracker:
    """Tracks formation history: for each final baryon, did a bound diquark
    (same-type attractive pair) persist before the third quark bound?

    The diquark-first fraction is the dynamical generalization of Lebed's
    static probability.
    """

    def __init__(self, window=1.0):
        self.window = window
        self.pair_bound_since = {}      # (i,j) same-type pair -> streak start t
        self.cluster_first_seen = {}    # members -> streak start t

    def update(self, t, edges, clusters, labels):
        # same-type (qq or qbar-qbar) bound pairs
        current_pairs = set()
        for i, j in edges:
            if (labels[i] < 3) == (labels[j] < 3):
                key = (min(i, j), max(i, j))
                current_pairs.add(key)
                self.pair_bound_since.setdefault(key, t)
        for key in list(self.pair_bound_since):
            if key not in current_pairs:
                del self.pair_bound_since[key]

        current_clusters = set()
        for cl in clusters:
            key = cl["members"]
            current_clusters.add(key)
            self.cluster_first_seen.setdefault(key, t)
        for key in list(self.cluster_first_seen):
            if key not in current_clusters:
                del self.cluster_first_seen[key]

    def diquark_first_fraction(self, final_clusters, species=("baryon", "antibaryon")):
        """Fraction of final (anti)baryons whose history shows an internal
        same-type pair bound >= window before the full cluster appeared."""
        n_total = 0
        n_diquark_first = 0
        for cl in final_clusters:
            if cl["species"] not in species:
                continue
            members = cl["members"]
            formed = self.cluster_first_seen.get(members)
            if formed is None:
                continue
            n_total += 1
            from itertools import combinations

            for i, j in combinations(members, 2):
                key = (min(i, j), max(i, j))
                since = self.pair_bound_since.get(key)
                if since is not None and since <= formed - self.window:
                    n_diquark_first += 1
                    break  # count each cluster at most once
        return (n_diquark_first / n_total if n_total else float("nan")), n_total


class PersistenceTracker:
    """Counts consecutive snapshots in which the identical constituent set
    appears; final tally keeps clusters seen >= n_persist times in a row."""

    def __init__(self, n_persist=3):
        self.n_persist = n_persist
        self.streaks = {}   # members -> consecutive count
        self.records = {}   # members -> last cluster dict

    def update(self, clusters):
        seen = set()
        for cl in clusters:
            key = cl["members"]
            seen.add(key)
            self.streaks[key] = self.streaks.get(key, 0) + 1
            self.records[key] = cl
        for key in list(self.streaks):
            if key not in seen:
                del self.streaks[key]

    def persistent_clusters(self):
        return [self.records[k] for k, n in self.streaks.items()
                if n >= self.n_persist]

    def tally(self):
        counts = Counter()
        exotic_counts = Counter()
        for cl in self.persistent_clusters():
            counts[cl["species"]] += 1
            if cl["exotic"]:
                exotic_counts[cl["species"]] += 1
        return dict(counts), dict(exotic_counts)
