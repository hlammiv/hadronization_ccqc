import numpy as np
import pytest

from diquark_md import color
from diquark_md.observables import pair_correlation

C_TABLE = color.build_c_table()


def test_ideal_gas_gr_flat():
    rng = np.random.default_rng(4)
    n, L = 3000, 10.0
    x = rng.uniform(0, L, (n, 3))
    labels = np.tile(np.arange(6), n // 6)[:n].astype(np.int64)
    alive = np.ones(n, dtype=np.bool_)
    r, g = pair_correlation(x, labels, alive, C_TABLE, L, r_max=2.0, n_bins=20)
    # class means flat at 1; per-bin scatter bounded by Poisson noise
    assert np.allclose(np.nanmean(g[:, 5:], axis=1), 1.0, atol=0.03)
    assert np.nanmax(np.abs(g[:, 5:] - 1.0)) < 0.3


def test_equilibrated_contact_ordering():
    """After interacting equilibration, the contact region (r < 0.4 fm,
    aggregated) is enhanced for the strongest attraction (matched q-qbar)
    and depleted for the strongest repulsion (like qq)."""
    from diquark_md.io import load_config
    from diquark_md.sim import Simulation

    cfg = load_config("configs/default.toml")
    cfg.update(n_pairs=1000, t_equil=5.0)
    sim = Simulation(cfg, seed=3)
    sim.equilibrate()
    s = sim.state
    r, g = pair_correlation(s["x"], s["labels"], s["alive"], C_TABLE,
                            sim.L, r_max=0.4, n_bins=1)
    contact = g[:, 0]
    assert contact[0] > 1.05          # matched q-qbar enhanced
    assert contact[3] < 0.97          # like qq depleted
    assert contact[0] > contact[1] > contact[2] > contact[3]
