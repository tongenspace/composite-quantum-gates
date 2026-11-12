# Numerical verification

`independent_checks.json` compares all 96 controls with direct matrix exponentials and checks unitarity, the common-duration-error mapping, and ideal virtual-Z completion. `scorbutus.json` verifies both switchback targets. `control_specifications.json` checks coefficient identities and separation of the selection and evaluation noise ensembles. `table_metrics.json` and the two CSV files give the performance-table calculations on the specified grids.

Reproduce these calculations with `python reproduce.py check`, `python reproduce.py scorbutus`, `python design/check_controls.py`, and `python reproduce.py tables`. `python reproduce.py figures` draws all twelve figures from the controls and supplied study data.

The [evaluation-grid study](../studies/grid_convergence/README.md) checks the mean infidelity on B0.15 for four fixed controls. Its `checks.json` records agreement with the publication means, direct matrix-exponential checks, and differences from the 3201x3201 midpoint reference. Recalculate it with `python reproduce.py grid-convergence`.
