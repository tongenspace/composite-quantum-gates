# Composite quantum gates simultaneously compensated for multiple errors

Controls, data, and reproduction code for the manuscript by **Hristo G. Tonchev and Nikolay V. Vitanov** (AR12946).

The calculations use coherent two-level dynamics. The performance measure is average gate infidelity. Full-precision pulse coefficients are supplied; reproducing the reported results does not require optimizing the controls.

## Installation

Use Python 3.11 or later in a virtual environment. For static benchmarks and all figures drawn from the supplied data:

```sh
python -m pip install -r requirements-static.txt
```

To recalculate the phase and time-dependent-detuning studies, install the full numerical environment:

```sh
python -m pip install -r requirements.txt
```

## Reproduction

Run these commands from the repository root:

```sh
python reproduce.py verify
python reproduce.py check
python reproduce.py scorbutus
python design/check_controls.py
python reproduce.py tables
python reproduce.py figures
python reproduce.py grid-convergence
```

`verify` checks the file hashes. `check` compares all 96 controls with independent matrix exponentials and verifies the duration-error mapping and ideal virtual-Z completion.

`scorbutus` reconstructs both switchback targets and recalculates their metrics. `tables` recalculates the two performance tables. `figures` draws all twelve figures, using the supplied ensemble data for the phase and detuning-noise figures. Results go to `outputs/`.

`grid-convergence` evaluates the fixed X5c, H5c, X15c, and H15c controls on midpoint grids from 8x8 to 3201x3201 over B0.15, using the same common-peak coordinates as the manuscript tables. The 801x801 means differ from the 3201x3201 reference by less than 0.0032% for all four controls. The [convergence figure, data, and editable TeX](studies/grid_convergence/README.md) are supplied with the calculation scripts.

For selected table entries:

```sh
python reproduce.py tables --only X9a X13c X13w H9c H13c H13w
```

To recalculate the implementation studies or derivative checks:

```sh
python reproduce.py phase
python reproduce.py edges
python reproduce.py derivatives
```

The phase calculation uses X9a, H9c, X15c, and H15w. The finite-edge and noise calculation uses X9a, X13c, X13w, H9c, H13c, and H13w. The latter requires substantially more computation than the static benchmarks.

## Repository contents

| Location | Contents |
|---|---|
| `manuscript/` | Manuscript TeX source, bibliography, figures, and clean PDF |
| `data/controls.json` | Full-precision controls in chronological order |
| `data/static_metrics.json` | Domain means, sampled maxima, threshold fractions, and numerical checks |
| `src/rectangular.py` | Two-level propagator and average gate infidelity |
| `scripts/` | Benchmarks, plots, and independent checks |
| `studies/phase_precision/` | Gaussian phase offsets and digital phase rounding |
| `studies/edges_noise/` | Finite edges, continuous detuning noise, and propagation checks |
| `studies/grid_convergence/` | Fixed-control convergence of the B0.15 mean, with figure and data |
| `design/` | Design objectives, constraints, search inputs, and coefficient checks |
| `figures/published/` | Figures used in the manuscript |
| `validation/` | Numerical verification results |

See [Model and units](docs/MODEL.md), [Figure and table reproduction](docs/RESULTS.md), [SCORBUTUS](docs/SCORBUTUS.md), and [Design methods](design/METHODS.md).

## Conventions

Controls store phases in radians, amplitudes `a_k = Omega_k/Omega_0`, and durations in units of `T`, where `Omega_0 T = pi`. The Hamiltonian is

```text
H_k = [a_k Omega_0 (1+epsilon) (cos(phi_k) sigma_x - sin(phi_k) sigma_y)
       - Delta sigma_z] / 2.
```

The same physical detuning acts during every segment. At common peak drive, `p = max(a_k)` and `Delta/Omega_0 = p delta_c`. This conversion is applied once per sequence. The Hadamard controls implement the `Rx(pi/2)` primitive; ideal virtual-Z rotations complete the gate.

For the paired 13- and 15-pulse controls, **c** denotes central accuracy and **w** denotes width. Use the identifiers in `data/controls.json` to select a control. [The methods specification](design/METHODS.md) distinguishes the objectives and constraints of each design class.

## Scope

Means and threshold fractions describe sampled error domains. Sampled maxima do not certify continuum bounds or global optimality. H5c uses rectangular pulses in design. The other Hadamard controls include the finite-edge model and use detuning noise in candidate selection. Subsequent fixed-coefficient noise tests use fresh Gaussian coefficients relative to selection; cutoff points in the bandwidth sweep share coefficients.

The finite-band `1/f` spectrum assumes a specified high-frequency continuation. The calculations include no multilevel leakage or dissipative dynamics, and the virtual-Z rotations remain ideal.

## Manuscript and citation

The manuscript and its bibliography can be compiled with a standard REVTeX installation:

```sh
cd manuscript
latexmk -pdf main.tex
```

The repository methods table has a separate editable TeX source in `design/design_methods_table.tex`. Compile it from that directory with `latexmk -pdf design_methods_table.tex`.

Use [CITATION.cff](CITATION.cff) when citing the materials. The repository is available at [GitHub](https://github.com/tongenspace/composite-quantum-gates).
