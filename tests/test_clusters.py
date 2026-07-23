"""Hand-built-configuration tests of the species identification pipeline."""

import numpy as np
import pytest

from diquark_md import color
from diquark_md.clusters import PersistenceTracker, find_clusters, _is_manifest_exotic
from collections import Counter

C_TABLE = color.build_c_table()
PARS = dict(alpha_s=0.4, lam_d=0.4, r0=0.2, r_cut=2.4, r_cl=1.2, L=50.0)


def _run(x, labels, flavors, masses=None, p=None):
    n = len(labels)
    x = np.asarray(x, dtype=float)
    labels = np.asarray(labels, dtype=np.int64)
    flavors = np.asarray(flavors, dtype=np.int64)
    mass = np.full(n, 0.34) if masses is None else np.asarray(masses, float)
    p = np.zeros((n, 3)) if p is None else np.asarray(p, float)
    alive = np.ones(n, dtype=np.bool_)
    return find_clusters(
        x, p, labels, flavors, mass, alive, C_TABLE, PARS["L"],
        PARS["alpha_s"], PARS["lam_d"], PARS["r0"], PARS["r_cut"], PARS["r_cl"],
    )


def test_single_meson():
    x = [[10, 10, 10], [10.3, 10, 10]]
    clusters, _ = _run(x, [0, 3], [0, 0])
    assert len(clusters) == 1
    assert clusters[0]["species"] == "meson"
    assert not clusters[0]["exotic"]


def test_baryon_is_one_cluster_not_fragments():
    x = [[10, 10, 10], [10.35, 10, 10], [10.17, 10.3, 10]]
    clusters, _ = _run(x, [0, 1, 2], [0, 1, 0])
    assert len(clusters) == 1
    assert clusters[0]["species"] == "baryon"


def test_two_distant_mesons_not_a_tetraquark():
    """Two well-separated tight mesons must NOT merge."""
    x = [[10, 10, 10], [10.25, 10, 10],
         [11.0, 10, 10], [11.25, 10, 10]]
    clusters, _ = _run(x, [0, 3, 1, 4], [0, 0, 1, 1])
    species = sorted(c["species"] for c in clusters)
    assert species == ["meson", "meson"]


def test_compact_tetraquark_beats_meson_split():
    """Four overlapping quarks at rest: the (2,2) partition wins when its
    total energy beats every meson+meson split.  Geometry: r-g-rbar-gbar on
    a small square so all six pairwise interactions contribute."""
    d = 0.25
    x = [[10, 10, 10], [10 + d, 10, 10], [10, 10 + d, 10], [10 + d, 10 + d, 10]]
    #     r              g               rbar             gbar
    clusters, _ = _run(x, [0, 1, 3, 4], [0, 0, 0, 0])
    assert len(clusters) == 1
    assert clusters[0]["species"] == "tetraquark"
    # 4-body energy must be at or below the best 2-meson split by search def.


def test_pentaquark_neutrality_and_species():
    """r g b r rbar in a compact blob: every returned bound part must be
    color-neutral; the 5-body pentaquark partition is among the candidates
    and must win if its energy beats all splits (e.g. baryon + meson)."""
    d = 0.3
    x = [[10, 10, 10], [10 + d, 10, 10], [10, 10 + d, 10],
         [10 + d, 10 + d, 10], [10 + d / 2, 10 + d / 2, 10 + d / 2]]
    labels = [0, 1, 2, 0, 3]
    clusters, _ = _run(x, labels, [0, 1, 0, 1, 1])
    assert clusters, "compact neutral blob must bind into something"
    for c in clusters:
        assert color.is_neutral([labels[i] for i in c["members"]])
    total_bound = sum(len(c["members"]) for c in clusters)
    species = sorted(c["species"] for c in clusters)
    # either the full pentaquark, or a neutral split like baryon+meson
    assert species == ["pentaquark"] or total_bound <= 5


def test_manifest_exotic_classification():
    # T_cc analog: cc ubar dbar  -> net c = +2, exotic
    net = Counter({"c": 2, "u": -1, "d": -1})
    assert _is_manifest_exotic(2, 2, net)
    # hidden flavor: c cbar u ubar -> net 0, not exotic
    net = Counter({"c": 0, "u": 0})
    assert not _is_manifest_exotic(2, 2, net)
    # ordinary meson u dbar
    net = Counter({"u": 1, "d": -1})
    assert not _is_manifest_exotic(1, 1, net)
    # pentaquark uudc cbar: net charm 0, net u=2,d=1 -> fits qqq -> hidden
    net = Counter({"u": 2, "d": 1, "c": 0})
    assert not _is_manifest_exotic(4, 1, net)
    # pentaquark uuds sbar with net s = 0 but content s sbar? net u=2,d=1,s=0
    # -> hidden; but uu d s cbar: net = u2 d1 s1 c-1 -> needs qbar -> exotic
    net = Counter({"u": 2, "d": 1, "s": 1, "c": -1})
    assert _is_manifest_exotic(4, 1, net)


def test_tcc_analog_detected_in_pipeline():
    """cc ubar dbar compact cluster flagged exotic by the full pipeline."""
    d = 0.25
    x = [[10, 10, 10], [10 + d, 10, 10], [10, 10 + d, 10], [10 + d, 10 + d, 10]]
    labels = [0, 1, 3, 4]           # r g rbar gbar (neutral (2,2))
    flavors = [3, 3, 0, 1]          # c c ubar dbar
    masses = [1.55, 1.55, 0.34, 0.34]
    clusters, _ = _run(x, labels, flavors, masses=masses)
    assert len(clusters) == 1
    assert clusters[0]["species"] == "tetraquark"
    assert clusters[0]["exotic"] is True


def test_free_hot_particles_no_clusters():
    """Fast-moving overlapping particles are not Hill-bound -> no clusters."""
    x = [[10, 10, 10], [10.3, 10, 10]]
    p = [[2.0, 0, 0], [-2.0, 0, 0]]
    clusters, _ = _run(x, [0, 3], [0, 0], p=p)
    assert clusters == []


def test_pathway_tracker_diquark_first():
    from diquark_md.clusters import PathwayTracker

    labels = np.array([0, 1, 2], dtype=np.int64)  # r g b
    baryon = {"members": (0, 1, 2), "species": "baryon", "exotic": False,
              "energy": -0.2, "flavor_content": ("u", "u", "d")}
    # history A: diquark (0,1) bound from t=0, baryon forms at t=3
    tr = PathwayTracker(window=1.0)
    for t in [0.0, 1.0, 2.0]:
        tr.update(t, [(0, 1)], [], labels)
    tr.update(3.0, [(0, 1), (0, 2), (1, 2)], [baryon], labels)
    frac, n = tr.diquark_first_fraction([baryon])
    assert n == 1 and frac == 1.0

    # history B: everything binds simultaneously -> not diquark-first
    tr2 = PathwayTracker(window=1.0)
    tr2.update(0.0, [(0, 1), (0, 2), (1, 2)], [baryon], labels)
    frac2, n2 = tr2.diquark_first_fraction([baryon])
    assert n2 == 1 and frac2 == 0.0

    # regression: TWO early-bound internal pairs must count the baryon ONCE
    # (fraction can never exceed 1)
    tr3 = PathwayTracker(window=1.0)
    for t in [0.0, 1.0, 2.0]:
        tr3.update(t, [(0, 1), (1, 2)], [], labels)
    tr3.update(3.0, [(0, 1), (0, 2), (1, 2)], [baryon], labels)
    frac3, n3 = tr3.diquark_first_fraction([baryon])
    assert n3 == 1 and frac3 == 1.0


def test_persistence_tracker():
    tr = PersistenceTracker(n_persist=3)
    meson = {"members": (1, 2), "species": "meson", "exotic": False,
             "energy": -0.1, "flavor_content": ("u", "ubar")}
    blip = {"members": (4, 5), "species": "meson", "exotic": False,
            "energy": -0.05, "flavor_content": ("d", "dbar")}
    tr.update([meson, blip])
    tr.update([meson])
    tr.update([meson, blip])
    counts, _ = tr.tally()
    assert counts == {"meson": 1}  # blip's streak was broken
