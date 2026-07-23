"""Validation bridge to Lebed, PRD 94, 034039 (2016), Tables 1-2 (hard wall).

Convention (locked analytically): hard-wall screening radius R; both
nearest-neighbor distributions truncated at R and normalized; diquark wins
iff r_qbar > sqrt(2) r_q; effective densities n_1 = n0/3, n_2 = n0/9 at
per-species density n0.
"""

import numpy as np
import pytest

from diquark_md.observables import lebed_analytic, lebed_static_mc


def test_analytic_low_density_limit():
    # Lebed Table 2: n0 = 1/(10R)^3 -> 17.69%
    assert lebed_analytic(1e-3) == pytest.approx(0.1769, abs=2e-3)


def test_analytic_table2_point():
    # Lebed Table 2: n0 = 1/R^3 -> 26.01%
    assert lebed_analytic(1.0) == pytest.approx(0.2601, abs=2e-3)


def test_analytic_reference_density():
    # Lebed Tables 1-2: n0 = (2/R)^3 -> 50.29%
    assert lebed_analytic(8.0) == pytest.approx(0.5029, abs=2e-3)


def test_analytic_asymptotic():
    # unscreened limit: n1/(k^3 n2 + n1) = 3/(3 + 2 sqrt2) = 51.47%
    assert lebed_analytic(1e3) == pytest.approx(3.0 / (3.0 + 2.0 * np.sqrt(2.0)), rel=1e-3)


def test_mc_matches_analytic_lebed_mode():
    """Ideal-gas MC with 1/3 / 1/9 thinning reproduces the analytic value."""
    rng = np.random.default_rng(42)
    n0 = 8.0  # per-species, units R = 1
    n_side = 12.0
    L = float(n_side)
    n_each = int(n0 * L**3)
    probs = []
    for _ in range(3):
        x = rng.uniform(0, L, (2 * n_each, 3))
        labels = np.concatenate([
            rng.integers(0, 3, n_each),      # quarks (colors irrelevant in lebed mode)
            rng.integers(3, 6, n_each),      # antiquarks
        ]).astype(np.int64)
        p, n_ev = lebed_static_mc(x, labels, L, R=1.0, mode="lebed", rng=rng)
        probs.append(p)
    p_mc = float(np.mean(probs))
    assert p_mc == pytest.approx(lebed_analytic(n0), abs=0.01)


def test_mc_discrete_mode_counting_shift():
    """Discrete color labels change the attractive-partner counting
    (n1 = 2nq/3, n2 = nqbar/3): quantifies the 2:1-vs-3:1 artifact."""
    rng = np.random.default_rng(7)
    n0 = 8.0
    L = 12.0
    n_each = int(n0 * L**3)
    x = rng.uniform(0, L, (2 * n_each, 3))
    labels = np.concatenate([
        np.repeat(np.arange(3), n_each // 3 + 1)[:n_each],
        np.repeat(np.arange(3, 6), n_each // 3 + 1)[:n_each],
    ]).astype(np.int64)
    p_disc, _ = lebed_static_mc(x, labels, L, R=1.0, mode="discrete", rng=rng)
    expected = lebed_analytic(n0, frac1=2.0 / 3.0, frac2=1.0 / 3.0)
    assert p_disc == pytest.approx(expected, abs=0.01)
    # and it differs measurably from the Lebed-mode value (physics point)
    assert abs(p_disc - lebed_analytic(n0)) > 0.02
