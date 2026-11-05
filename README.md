# Code and data for "Erosion by a temperate glacier: erosion and sediment export depending on ice dynamics, bed geometry and meltwater flux"

F. Herman and F. De Doncker. This repository holds the model, the scripts that make every figure and table of the paper, and the data they read.

## Requirements

Python 3.9 or later with `numpy`, `scipy` and `matplotlib` (`pip install -r requirements.txt`). Everything runs on a laptop. The first Blatter--Pattyn solution of each glacier takes about a minute and is cached in `_cache/`.

## Layout

| File | Content |
|---|---|
| `blatter_flow.py` | Blatter--Pattyn flowline solver, hydrology, supercooling, frozen fringe, till and sediment routing, steady-state diagnostics |
| `transient.py`, `transient_sediment.py` | Evolving ice geometry and sediment reservoirs with the fresh/stored tracer |
| `run_transient.py` | One transient run, `python run_transient.py arolla\|franzjosef\|gorner [options]` |
| `run_all_transient.sh` | All transient runs of the paper |
| `fig*.py`, `table*.py`, `sweep_discharge.py`, `sweep_parameters.py`, `monte_carlo.py`, `calibrate_kg.py`, `invert_*.py` | Figures, tables and numbers, see below |
| `tests/test_blatter_flow.py` | Analytical tests of the solver and the sediment routing |
| `data/` | Input data, see below |
| `figures/`, `output/` | Written by the scripts |

## Reproducing the paper

Run the scripts from this directory. Figures are written to `figures/`, transient runs to `output/`, and tables and numbers are printed.

| Paper | Command | Notes |
|---|---|---|
| Fig. 3 | `python fig3_verification.py` | solver verification and velocities |
| Figs. 4 and 5 | `python fig4_5_diagnostics.py` | steady state at 63 um and 10 mm; prints the sediment budgets |
| Fig. 6 | `python fig6_export.py` | modelled erosion and export against the observed discharge |
| Fig. 7 | `python fig7_fj_spatial.py` | Franz Josef Glacier along the flowline |
| Fig. 8 | `python fig8_gorner_spatial.py` | Gornergletscher along the flowline |
| Fig. 9 | `python fig9_grain_size.py` | export against grain size |
| Fig. 10 and Section 4.5 | `./run_all_transient.sh`, then `python fig10_transient.py` | the transient runs take about 70 minutes with four runs in parallel; the figure script prints the numbers quoted in Section 4.5 |
| Table 2 | `python table2_mixture.py` | mixture of grain sizes |
| Table 3 | `python table3_naive_kg.py` | bulk estimate of the proportionality constant |
| Proportionality constants | `python calibrate_kg.py` | K_g for l = 1, 2 and 3, reproduces `K_G_BY_LOADER` in `blatter_flow.py` |
| Sections 4.1 and 4.3 (Franz Josef Glacier) | `python invert_fj_velocity.py` | inversion of C_s(x) and f(x) and the regression of the mapped erosion |
| Section 5.3 and Appendix C | `python sweep_discharge.py` | transfer ratio against discharge and the budget residual |
| Section 4.6 and Fig. 11 | `python sweep_parameters.py`, `python monte_carlo.py run`, `python monte_carlo.py transient`, `python monte_carlo.py control`, `python monte_carlo.py summary` | one-at-a-time sweep (10 min), 200 Latin hypercube draws (30 min), 30 transient runs with and without forcing (1 h each), then the percentiles, rank correlations and figure; times on eight cores |
| Tests | `python tests/test_blatter_flow.py --slow` | |

Figures 1 and 2 are drawings and are not produced by the code. The parameters of Table 4 are the constants at the top of `blatter_flow.py` and `transient.py`.

## Data

| File | Content and source |
|---|---|
| `data/arolla.dat` | Haut Glacier d'Arolla flowline: distance, bed and surface elevation (Blatter 1995) |
| `data/franzjosef/Bedrock_Anderson2014.csv` | Franz Josef Glacier bed and surface profiles (Anderson et al. 2014) |
| `data/franzjosef/herman2015/` | Mapped erosion rates and glacier width (Herman et al. 2015), see the `README.md` there |
| `data/franzjosef/surface_velocity_2013.csv` | Observed surface speed against distance to the sampling station (Herman et al. 2015) |
| `data/gorner/gorner.dat` | Gornergletscher flowline: distance, bed and surface elevation (De Doncker et al. 2026) |
| `data/gorner/gorner_flowline.csv` | Along the flowline, the 2018 surface speed from Planet imagery, its uncertainty, and the provenance-derived erosion rate (De Doncker et al. 2026) |
| `data/inverted/*.npy` | C_s(x) and f(x) used in the paper (rows C_s and f) |

The inversion scripts write their result to `output/`. Rerunning `invert_fj_velocity.py` gives the coefficients of `data/inverted/` exactly. Rerunning `invert_gorner_velocity.py` gives coefficients with the same misfit (a factor of 1.36 between modelled and observed speed) but not identical values, because the inversion of a speed profile alone does not fix C_s(x) and f(x) uniquely. The numbers of the paper use the files in `data/inverted/`.

## Citation

Please cite the paper and the archived version of this repository (DOI to be added).

## Licence

The code is released under the MIT licence (`LICENSE`). The input data come from the publications listed above and keep their original terms; please cite those sources when you use them.
