"""Discrete color labels, channel factors, and neutrality combinatorics.

Labels: 0,1,2 = r,g,b (quarks); 3,4,5 = rbar,gbar,bbar (antiquarks).

The pair coupling is the diagonal expectation value of the one-gluon-exchange
operator in the definite-color product state,

    C_ij = 2 <c_i c_j| F_1.F_2 |c_i c_j>,

computed from the SU(3) Fierz identity sum_a T^a_ii T^a_kk = (delta_ik - 1/3)/2:

    qq  same color   (rr):    +2/3   (pure 6, repulsive)
    qq  diff color   (rg):    -1/3   (1/2 3bar + 1/2 6, attractive)
    qqbar matched    (r rbar): -2/3  (1/3 1 + 2/3 8, attractive)
    qqbar mismatched (r gbar): +1/3  (pure 8, repulsive)

Full-strength channel constants (Lebed's C = C2(R) - C2(R1) - C2(R2)):
singlet -8/3, 3bar -4/3, 6 +2/3, 8 +1/3.  The potential is
V = (C/2) alpha_s hbar-c f(r), so a singlet qqbar gives -(4/3) alpha_s/r.
"""

from itertools import combinations

import numpy as np

N_LABELS = 6

# Full-strength channel Casimir combinations
C_SINGLET = -8.0 / 3.0
C_3BAR = -4.0 / 3.0
C_6 = 2.0 / 3.0
C_8 = 1.0 / 3.0


def is_quark(label: int) -> bool:
    return label < 3


def anticolor_of(label: int) -> int:
    """Matching anticolor label for a quark label (and vice versa)."""
    return (label + 3) % 6


def _mean_channel_factor(l1: int, l2: int) -> float:
    q1, q2 = l1 < 3, l2 < 3
    if q1 == q2:  # qq or qbar-qbar
        return 2.0 / 3.0 if (l1 % 3) == (l2 % 3) else -1.0 / 3.0
    # q-qbar
    return -2.0 / 3.0 if (l1 % 3) == (l2 % 3) else 1.0 / 3.0


def _max_channel_factor(l1: int, l2: int) -> float:
    """Variant: assign each pair its most attractive accessible channel at full
    strength (systematics bracket; see paper App. A)."""
    q1, q2 = l1 < 3, l2 < 3
    if q1 == q2:
        return C_6 if (l1 % 3) == (l2 % 3) else C_3BAR
    return C_SINGLET if (l1 % 3) == (l2 % 3) else C_8


def build_c_table(color_mode: str = "mean_channel") -> np.ndarray:
    """6x6 table of pair coupling factors C_ij indexed by labels."""
    if color_mode == "mean_channel":
        f = _mean_channel_factor
    elif color_mode == "max_channel":
        f = _max_channel_factor
    else:
        raise ValueError(f"unknown color_mode {color_mode!r}")
    table = np.empty((N_LABELS, N_LABELS), dtype=np.float64)
    for a in range(N_LABELS):
        for b in range(N_LABELS):
            table[a, b] = f(a, b)
    return table


def color_vector(labels) -> np.ndarray:
    """Net color weight vector (n_r - n_rbar, n_g - n_gbar, n_b - n_bbar)."""
    labels = np.asarray(labels)
    v = np.zeros(3, dtype=np.int64)
    for c in range(3):
        v[c] = int(np.sum(labels == c)) - int(np.sum(labels == c + 3))
    return v


def is_neutral(labels) -> bool:
    """A multiset of labels contains a color singlet in its decomposition iff
    all three color-weight differences are equal (indices fully contractible
    with deltas and epsilons)."""
    v = color_vector(labels)
    return v[0] == v[1] == v[2]


def neutral_partitions(labels):
    """Yield all partitions of range(len(labels)) into blocks that are each
    either singletons (free particles) or color-neutral multi-particle sets.

    Yields lists of tuples of indices.  Exponential in len(labels); intended
    for pre-clusters of size <= ~12 (heavily pruned by the neutrality
    requirement on blocks).
    """
    labels = list(labels)
    n = len(labels)

    def rec(remaining):
        if not remaining:
            yield []
            return
        first = remaining[0]
        rest = remaining[1:]
        # first as a free singleton
        for sub in rec(rest):
            yield [(first,)] + sub
        # first in a neutral block of size >= 2
        for size in range(1, len(rest) + 1):
            for combo in combinations(rest, size):
                block = (first,) + combo
                if is_neutral([labels[i] for i in block]):
                    leftover = [i for i in rest if i not in combo]
                    for sub in rec(leftover):
                        yield [block] + sub

    yield from rec(list(range(n)))


def species_of(labels) -> str:
    """Species name for a bound neutral cluster from its (n_q, n_qbar)."""
    labels = np.asarray(labels)
    nq = int(np.sum(labels < 3))
    nqbar = int(np.sum(labels >= 3))
    key = (nq, nqbar)
    named = {
        (1, 1): "meson",
        (3, 0): "baryon",
        (0, 3): "antibaryon",
        (2, 2): "tetraquark",
        (4, 1): "pentaquark",
        (1, 4): "antipentaquark",
        (6, 0): "dibaryon",
        (0, 6): "antidibaryon",
        (3, 3): "hexaquark",
    }
    return named.get(key, f"exotic_{nq}q{nqbar}qbar")
