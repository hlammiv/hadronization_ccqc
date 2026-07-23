"""Constants and derived plasma parameters.

Internal units: hbar = c = 1, lengths in fm, energies in GeV, time in fm/c.
"""

import math

HBARC = 0.19733  # GeV fm

# Constituent quark masses (GeV)
FLAVOR_NAMES = ("u", "d", "s", "c")
FLAVOR_MASSES = {"u": 0.34, "d": 0.34, "s": 0.48, "c": 1.55}
FLAVOR_IDS = {name: i for i, name in enumerate(FLAVOR_NAMES)}


def mass_of(flavor_id: int) -> float:
    return FLAVOR_MASSES[FLAVOR_NAMES[flavor_id]]


def wigner_seitz_radius(n: float) -> float:
    """a_ws = (3 / 4 pi n)^(1/3) for total number density n [fm^-3]."""
    return (3.0 / (4.0 * math.pi * n)) ** (1.0 / 3.0)


def coupling_gamma(alpha_s: float, n: float, T: float) -> float:
    """Yukawa-plasma coupling parameter Gamma = alpha_s hbar-c / (a_ws T)."""
    return alpha_s * HBARC / (wigner_seitz_radius(n) * T)


def screening_kappa(n: float, lam_d: float) -> float:
    """kappa = a_ws / lambda_D."""
    return wigner_seitz_radius(n) / lam_d


def bessel_k2(x: float) -> float:
    """Modified Bessel function K_2(x) via the integral representation.

    K_2(x) = int_0^inf exp(-x cosh t) cosh(2t) dt.  Adequate for the
    thermal-weight ratios used here (x = m/T in [1, 12]); relative accuracy
    ~1e-10 with the fixed grid below.
    """
    if x <= 0:
        raise ValueError("bessel_k2 requires x > 0")
    # cosh grows fast: t up to ~ acosh(700/x) covers double range
    t_max = math.acosh(max(700.0 / x, 2.0))
    n_pts = 2000
    dt = t_max / n_pts
    total = 0.0
    for i in range(n_pts + 1):
        t = i * dt
        w = 0.5 if (i == 0 or i == n_pts) else 1.0
        total += w * math.exp(-x * math.cosh(t)) * math.cosh(2.0 * t)
    return total * dt


def thermal_flavor_weights(T: float, flavors=("u", "d", "s")) -> dict:
    """Relative equilibrium pair densities n_f ~ m_f^2 T K_2(m_f/T) (Boltzmann)."""
    w = {}
    for f in flavors:
        m = FLAVOR_MASSES[f]
        w[f] = m * m * T * bessel_k2(m / T)
    s = sum(w.values())
    return {f: v / s for f, v in w.items()}
