import numpy as np
import pytest

from diquark_md import color


def test_mean_channel_values():
    t = color.build_c_table("mean_channel")
    # qq same / different color
    assert t[0, 0] == pytest.approx(2.0 / 3.0)
    assert t[0, 1] == pytest.approx(-1.0 / 3.0)
    # qbar-qbar mirrors qq
    assert t[3, 3] == pytest.approx(2.0 / 3.0)
    assert t[3, 4] == pytest.approx(-1.0 / 3.0)
    # q-qbar matched / mismatched
    assert t[0, 3] == pytest.approx(-2.0 / 3.0)
    assert t[0, 4] == pytest.approx(1.0 / 3.0)
    # symmetry
    assert np.allclose(t, t.T)


def test_trace_consistency():
    """Summed over the 9 color combinations, the mean-channel factors vanish
    for both qq and q-qbar (matches the exact channel decomposition)."""
    t = color.build_c_table("mean_channel")
    assert np.sum(t[:3, :3]) == pytest.approx(0.0, abs=1e-14)
    assert np.sum(t[:3, 3:]) == pytest.approx(0.0, abs=1e-14)


def test_strength_ratio_preserved():
    """Strongest qqbar attraction : strongest qq attraction = 2 : 1
    (Lebed's k = sqrt(2) criterion carries over)."""
    t = color.build_c_table("mean_channel")
    assert t[0, 3] / t[0, 1] == pytest.approx(2.0)


def test_neutrality():
    r, g, b, rb, gb, bb = 0, 1, 2, 3, 4, 5
    assert color.is_neutral([r, rb])            # meson
    assert not color.is_neutral([r, gb])
    assert color.is_neutral([r, g, b])          # baryon
    assert not color.is_neutral([r, g, g])
    assert color.is_neutral([rb, gb, bb])       # antibaryon
    assert color.is_neutral([r, g, rb, gb])     # tetraquark
    assert color.is_neutral([r, g, b, r, rb])   # pentaquark
    assert not color.is_neutral([r, g, b, r])


def test_neutral_partitions_baryon_plus_meson():
    # rgb + (r rbar): partitions must include {baryon, meson} split
    labels = [0, 1, 2, 0, 3]
    parts = list(color.neutral_partitions(labels))
    as_sets = [sorted(tuple(sorted(b)) for b in p) for p in parts]
    assert sorted([(0, 1, 2), (3, 4)]) in as_sets
    # all-singletons always present
    assert sorted([(0,), (1,), (2,), (3,), (4,)]) in as_sets
    # every multi-particle block in every partition is neutral
    for p in parts:
        for block in p:
            if len(block) > 1:
                assert color.is_neutral([labels[i] for i in block])


def test_species_names():
    assert color.species_of([0, 3]) == "meson"
    assert color.species_of([0, 1, 2]) == "baryon"
    assert color.species_of([3, 4, 5]) == "antibaryon"
    assert color.species_of([0, 1, 3, 4]) == "tetraquark"
    assert color.species_of([0, 1, 2, 0, 3]) == "pentaquark"
