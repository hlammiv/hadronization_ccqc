import numpy as np
import pytest

from diquark_md import color
from diquark_md.io import load_config
from diquark_md.potentials import pair_potential, pair_separation
from diquark_md.sim import Simulation
from diquark_md.units import FLAVOR_IDS


def test_cc_diquark_seeding():
    cfg = load_config("configs/default.toml")
    cfg.update(n_pairs=150, n_ccbar=6, seed_cc_diquarks=True)
    sim = Simulation(cfg, seed=5)
    s = sim.state
    labels, flavors, x, p = s["labels"], s["flavors"], s["x"], s["p"]

    # global invariants intact
    assert color.is_neutral(labels)
    c_q = np.where((flavors == FLAVOR_IDS["c"]) & (labels < 3))[0]
    assert len(c_q) == 6

    # planted pairs: close, distinct colors (attractive channel), bound
    c_table = color.build_c_table()
    for a, b in zip(c_q[0::2], c_q[1::2]):
        r = pair_separation(x, int(a), int(b), sim.L)
        assert r == pytest.approx(cfg.get("cc_seed_separation", 0.15), abs=1e-9)
        assert labels[a] != labels[b]
        assert c_table[labels[a], labels[b]] < 0
        # zero relative momentum + attractive potential => bound pair
        assert np.allclose(p[a], p[b])
        v = pair_potential(r, c_table[labels[a], labels[b]], cfg["alpha_s"],
                           cfg["lam_d0"], cfg["r0"], 6 * cfg["lam_d0"])
        assert v < 0


def test_fixed_screening_flag():
    cfg = load_config("configs/default.toml")
    cfg.update(n_pairs=48, t_equil=0.5, a_max=1.5, comoving_screening=False,
               lambda_scatt=0.0, p_gamma=0.0)
    sim = Simulation(cfg, seed=1)
    sim.equilibrate()
    sim.run_production()
    assert sim.lam_d == pytest.approx(cfg["lam_d0"])

    cfg["comoving_screening"] = True
    sim2 = Simulation(cfg, seed=1)
    sim2.equilibrate()
    sim2.run_production()
    assert sim2.lam_d == pytest.approx(cfg["lam_d0"] * sim2.a)
