# diquark_stuff

Dynamical extension of R. F. Lebed, *"How Often Do Diquarks Form? A Very Simple
Model"*, PRD **94**, 034039 (2016) [arXiv:1606.07108], plus a renormalizable
operator framework for extracting the same hadron-species ratios from a
quantum lattice-QCD simulation.

Two components:

- `src/diquark_md/` — classical molecular dynamics of a quark–antiquark gas
  with discrete color labels, Casimir-weighted screened-Coulomb interactions,
  Hubble expansion/freeze-out, Poisson-hazard annihilation (photon and gluon
  channels, tunable timescales), thermal flavors (u, d, s + injected c c̄),
  and color-singlet cluster identification of
  meson : baryon : tetraquark : pentaquark yields.
- `paper/` — RevTeX 4.2 manuscript (PRD target): classical results + the
  quantum operator/measurement framework (charge counting, spectral ID,
  asymptotic pair spectra, flowed coalescence projectors).

## Quick start

```bash
pip install -e .[dev]
pytest                                     # physics + unit tests
python -m diquark_md run configs/static_lebed.toml   # reproduce Lebed's static P ~ 50%
python -m diquark_md run configs/default.toml        # one full expanding-fireball event
```

Outputs land in `runs/` as HDF5 with the resolved config and RNG seed archived.
Figures: `scripts/plot_*.py`.

## Parallelism and performance

Two independent levels:

- **Across events**: scan configs (`mode = "scan"`) fan (point × seed) jobs
  over `max_workers` processes; each worker uses `numba_threads` threads
  (both set in the scan TOML). Total cores ≈ `max_workers × numba_threads`;
  budget ~300 MB RAM per worker. On a constrained machine use e.g.
  `max_workers = 4`, `numba_threads = 2`.
- **Within an event**: force/neighbor kernels are Numba-parallel
  (`NUMBA_NUM_THREADS` env for single runs).

Forces, reaction candidates, and cluster edges all run over a linked-cell +
Verlet-skin neighbor list (`neighbors.py`, CSR layout; exact vs brute force
by test). Amortized speedups vs the all-pairs kernels: ~3.5× at N = 3000,
~7.8× at N = 10⁴ for forces, and the reaction candidate scan drops from
O(N²) *serial* to sub-ms. Pure Hubble rescaling never triggers a rebuild
(the covered radius scales with the box).

## Units

ħ = c = 1; lengths in fm, energies in GeV, time in fm/c; ħc = 0.19733 GeV·fm.
