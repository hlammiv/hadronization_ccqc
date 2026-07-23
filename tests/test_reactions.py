import math

import numpy as np
import pytest

from diquark_md import color
from diquark_md.reactions import (
    ReactionEngine,
    find_candidate_pairs,
    make_injected_pair,
    thermal_flavor_probs,
)

C_TABLE = color.build_c_table()
POT_PARS_TEMPLATE = dict(alpha_s=0.4, lam_d=0.4, r0=0.2)


def _params(**over):
    p = dict(lambda_scatt=1.0, tau_bound=math.inf, p_gamma=0.5,
             t_chem=0.16, d_ann=0.2)
    p.update(over)
    return p


def _state_pair(r_sep, p_rel=0.5, flavor=0, L=20.0):
    """A single matched u-ubar pair at separation r_sep, relative momentum
    p_rel along x (each +-p_rel/2)."""
    x = np.array([[10.0, 10.0, 10.0], [10.0 + r_sep, 10.0, 10.0]])
    p = np.array([[p_rel / 2, 0.0, 0.0], [-p_rel / 2, 0.0, 0.0]])
    return dict(
        x=x, p=p,
        labels=np.array([0, 3], dtype=np.int64),
        flavors=np.array([flavor, flavor], dtype=np.int64),
        mass=np.array([0.34, 0.34]),
        alive=np.ones(2, dtype=np.bool_),
    ), L


def _pot_pars(L):
    return (C_TABLE, 0.4, 0.4, 0.2, min(L / 2, 2.4), L)


def test_candidate_pairs_matching_only():
    state, L = _state_pair(0.1)
    # mismatch the colors -> no candidates
    state["labels"] = np.array([0, 4], dtype=np.int64)
    ii, jj, rr = find_candidate_pairs(
        state["x"], state["labels"], state["flavors"], state["alive"], L, 0.2
    )
    assert len(ii) == 0
    # mismatch the flavors -> no candidates
    state["labels"] = np.array([0, 3], dtype=np.int64)
    state["flavors"] = np.array([0, 2], dtype=np.int64)
    ii, jj, rr = find_candidate_pairs(
        state["x"], state["labels"], state["flavors"], state["alive"], L, 0.2
    )
    assert len(ii) == 0
    # matched -> one candidate
    state["flavors"] = np.array([0, 0], dtype=np.int64)
    ii, jj, rr = find_candidate_pairs(
        state["x"], state["labels"], state["flavors"], state["alive"], L, 0.2
    )
    assert len(ii) == 1 and rr[0] == pytest.approx(0.1)


def test_scattering_hazard_rate():
    """Unbound overlapping pair annihilates at rate lambda_scatt/3 * branch:
    survival over time t is exp(-rate t), checked against ensemble MC.
    Same expected count at dt and dt/2 (Poisson hazard is dt-independent)."""
    lam = 2.0
    t_total = 1.0
    for dt in (0.01, 0.005):
        rng = np.random.default_rng(11)
        n_trials = 2000
        survived = 0
        for _ in range(n_trials):
            state, L = _state_pair(0.05, p_rel=2.0)  # hot -> unbound
            eng = ReactionEngine(_params(lambda_scatt=lam, p_gamma=1.0), rng)
            dead = False
            for step in range(int(t_total / dt)):
                if eng.step(state, step * dt, dt, t_eff=0.3,
                            potential_pars=_pot_pars(L)):
                    dead = True
                    break
            if not dead:
                survived += 1
        expected = math.exp(-lam / 3.0 * t_total)
        assert survived / n_trials == pytest.approx(expected, abs=0.03), f"dt={dt}"


def test_bound_pair_lifetime_and_immortal_limit():
    """Deeply bound overlapping pair dies at rate 1/(3 tau_bound); with
    tau_bound = inf it is immortal (original design recovered)."""
    # immortal limit
    rng = np.random.default_rng(5)
    state, L = _state_pair(0.05, p_rel=0.01)  # cold, overlapping -> bound
    eng = ReactionEngine(_params(tau_bound=math.inf, p_gamma=1.0), rng)
    for step in range(2000):
        assert eng.step(state, step * 0.01, 0.01, t_eff=0.05,
                        potential_pars=_pot_pars(L)) == 0
    assert state["alive"].all()

    # finite lifetime: survival matches exp(-t/(3 tau))
    tau = 1.0
    t_total = 2.0
    dt = 0.01
    rng = np.random.default_rng(6)
    survived = 0
    n_trials = 1500
    for _ in range(n_trials):
        state, L = _state_pair(0.05, p_rel=0.01)
        eng = ReactionEngine(_params(tau_bound=tau, p_gamma=1.0), rng)
        dead = False
        for step in range(int(t_total / dt)):
            if eng.step(state, step * dt, dt, t_eff=0.05,
                        potential_pars=_pot_pars(L)):
                dead = True
                break
        if not dead:
            survived += 1
    expected = math.exp(-t_total / (3.0 * tau))
    assert survived / n_trials == pytest.approx(expected, abs=0.04)


def test_gluon_channel_gates_off_cold():
    """Below T_chem with p_gamma = 0 nothing can happen."""
    rng = np.random.default_rng(7)
    state, L = _state_pair(0.05, p_rel=2.0)
    eng = ReactionEngine(_params(lambda_scatt=50.0, p_gamma=0.0), rng)
    for step in range(500):
        assert eng.step(state, step * 0.01, 0.01, t_eff=0.10,
                        potential_pars=_pot_pars(L)) == 0
    assert state["alive"].all()


def test_photon_ledger_and_conservation():
    """Photon events conserve the pair four-momentum in the ledger."""
    rng = np.random.default_rng(8)
    state, L = _state_pair(0.05, p_rel=2.0)
    e_before = sum(math.sqrt(state["p"][i] @ state["p"][i] + state["mass"][i] ** 2)
                   for i in range(2))
    p_before = state["p"].sum(axis=0).copy()
    eng = ReactionEngine(_params(lambda_scatt=1e3, p_gamma=1.0), rng)
    for step in range(200):
        if eng.step(state, step * 0.01, 0.01, t_eff=0.3,
                    potential_pars=_pot_pars(L)):
            break
    assert not state["alive"].any()
    assert len(eng.photon_ledger) == 1
    t, e, px, py, pz = eng.photon_ledger[0]
    assert e == pytest.approx(e_before)
    assert np.allclose([px, py, pz], p_before)


def test_reinjection_conserves_color_flavor_energy():
    """Gluon-channel re-injection: color stays neutral, four-momentum is
    carried by the new pair, flavor may change but net flavor stays zero."""
    rng = np.random.default_rng(9)
    state, L = _state_pair(0.05, p_rel=2.0)
    e_before = sum(math.sqrt(state["p"][i] @ state["p"][i] + state["mass"][i] ** 2)
                   for i in range(2))
    p_before = state["p"].sum(axis=0).copy()
    eng = ReactionEngine(_params(lambda_scatt=1e3, p_gamma=0.0), rng)
    fired = 0
    for step in range(200):
        fired = eng.step(state, step * 0.01, 0.01, t_eff=0.3,
                         potential_pars=_pot_pars(L))
        if fired:
            break
    assert fired == 1 and eng.n_reinjections == 1
    alive = state["alive"]
    assert alive.sum() == 2
    # color neutral
    assert color.is_neutral(state["labels"][alive])
    # net flavor zero (pair is same-flavor q + qbar)
    labs = state["labels"][alive]
    flavs = state["flavors"][alive]
    assert flavs[labs < 3].tolist() == flavs[labs >= 3].tolist()
    # four-momentum carried over exactly
    e_after = sum(
        math.sqrt(state["p"][k] @ state["p"][k] + state["mass"][k] ** 2)
        for k in np.where(alive)[0]
    )
    p_after = state["p"][alive].sum(axis=0)
    assert e_after == pytest.approx(e_before, rel=1e-12)
    assert np.allclose(p_after, p_before)


def test_thermal_flavor_threshold():
    """Charm only opens when M_inv >= 2 m_c; strange suppressed vs light."""
    names, probs = thermal_flavor_probs(0.2, m_inv=1.0)
    assert "c" not in names and "s" in names
    d = dict(zip(names, probs))
    assert d["u"] > d["s"]
    names2, probs2 = thermal_flavor_probs(0.2, m_inv=4.0)
    assert "c" in names2
    d2 = dict(zip(names2, probs2))
    assert d2["c"] < d2["s"] < d2["u"]


def test_injected_pair_kinematics():
    """Injected pair: invariant mass preserved, lab four-momentum matches."""
    rng = np.random.default_rng(10)
    p_tot = np.array([0.3, -0.2, 0.5])
    m_new = 0.48
    m_inv = 1.5
    e_tot = math.sqrt(m_inv**2 + p_tot @ p_tot)
    x1, x2, p1, p2 = make_injected_pair(m_new, p_tot, e_tot, 0.2, 10.0, rng)
    e1 = math.sqrt(p1 @ p1 + m_new**2)
    e2 = math.sqrt(p2 @ p2 + m_new**2)
    assert np.allclose(p1 + p2, p_tot, atol=1e-12)
    assert e1 + e2 == pytest.approx(e_tot, rel=1e-12)
    # separation d_ann
    d = x2 - x1
    d -= 10.0 * np.round(d / 10.0)
    assert np.linalg.norm(d) == pytest.approx(0.2)
