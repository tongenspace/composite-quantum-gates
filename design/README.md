# Control design

The numerical specifications are in [METHODS.md](METHODS.md). Full-precision pulse coefficients are in `../data/controls.json`. The search code, initial vectors, constraints, and numerical solutions are supplied under `searches/`.

`coefficient_matches.json` maps each numerical control to exact matching records. The match checks pulse count, rotation angle, amplitudes, durations, and phases modulo 2 pi:

```sh
python design/check_controls.py
```

## Optional optimization runs

The published coefficients reproduce all reported results without optimization. To prepare an isolated search workspace:

```sh
python design/run_search.py x-long --pulse-count 13 --prepare-only
python design/run_search.py hadamard --pulse-count 13 --prepare-only
```

Omit `--prepare-only` to execute a search. Outputs remain under `outputs/`; the supplied control registry is not replaced. Optimization may converge to different feasible solutions. Objective values and candidate-selection criteria are separate quantities where noise is used in selection.

## Evaluation-grid convergence

The fixed-control resolution check is in [studies/grid_convergence](../studies/grid_convergence/README.md). Run `python reproduce.py grid-convergence` to recalculate the mean infidelity of X5c, H5c, X15c, and H15c on successively finer midpoint grids over B0.15. This calculation changes the evaluation mesh while keeping the pulse coefficients fixed.

## Methods table

Compile `design_methods_table.tex` from this directory with `latexmk -pdf design_methods_table.tex`. Its body is in `design_methods_table_body.tex`. The table distinguishes genuinely different objectives, error coordinates, constraints, and candidate-selection rules.
