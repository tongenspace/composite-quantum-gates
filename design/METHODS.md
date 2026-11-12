# Design objectives and constraints

The pulse coefficients in `data/controls.json` define the controls used in the manuscript. `coefficient_matches.json` identifies their corresponding search inputs and solutions by numerical equality, with phases compared modulo 2 pi. Run `python design/check_controls.py` to check these identities and the noise ensembles.

## Design classes

| Controls | Objective and selection |
|---|---|
| X5c | Smooth threshold coverage on the native B0.15 square, subject to nominal-infidelity, domain-mean, area, and peak bounds. The specified control increases coverage and reduces nominal error relative to its reference control; a unique scalar ranking or tie-break over all feasible candidates is unspecified. |
| X7c | Equal-weight smooth coverage on native and common-peak B0.5, with exact nominal matching and central-mean budgets. Selection ranks the smaller relative coverage gain and requires fine-grid and Sobol agreement. |
| X11c | The X7c coverage criterion, with additional constraints on the central sampled maxima. |
| X9c | Equal-weight native and common-peak central means, with exact nominal matching and resource bounds. Feasible candidates are ranked by the mean on a 401-point midpoint mesh. |
| X13c, X15c | The two-coordinate mean objective, exact target, amplitudes in [0.05,2], and area/pi at most N. Feasible polished candidates are ranked by mean and checked on finer grids and Sobol samples. |
| X13w | Saturated threshold loss on common-peak B0.5, exact target, paired area/peak caps, and sampled protection on B0.20. Selection specifies the largest checked connected threshold region; a tie-break is unspecified. |
| X15w | The width objective on common-peak B0.5, exact target, paired resource bounds, and protection on B0.20. Candidates are ranked by connected coverage and checked for feasibility on finer samples. |
| H5c | Rectangular pulses only. Two-coordinate mean refinement followed by smooth native B0.5 coverage, under nominal-target, resource, and mean-error constraints. Selection uses the largest qualifying fine-grid native coverage. |
| H7c, H11c | Weighted rectangular/finite-edge width objective after mean refinement. Selection maximizes 0.3 rectangular coverage + 0.7 noisy-edge coverage within the resource and central-mean limits. |
| H9c, H13c, H15c | Deterministic 0.3 rectangular + 0.7 finite-edge central mean, with exact nominal target and resource caps. Candidate selection uses the corresponding rectangular/noisy-edge mean. |
| H13w | Weighted width objective with paired resource bounds and two-model protection on B0.15. Selection uses weighted rectangular/noisy-edge coverage. |
| H15w | Weighted width objective with paired resource bounds and protection on B0.20, followed by finer central-constraint refinement. The feasible refined control is retained and its noisy coverage evaluated. |

`searches/` supplies the numerical objectives, smoothing parameters, grids, seeds, starting vectors, and stopping criteria. Each control is matched to its own specification; the suffix alone does not identify an optimization procedure. `design_methods_table.tex` provides the same classification in table form.

## Physical bounds


All numerical segments have equal duration T, Omega0 T = pi. Resource pairs below
are (sum a, max a), in that storage convention. The physical plots instead use
a common peak drive and delta_c = Delta/Omega_max; native delta = p delta_c.

| Control | Maximum area/pi | Maximum a | Minimum a |
|---|---:|---:|---:|
| X5c | 6.9933 | 2 | 1e-5 |
| X7c | 6.9583 | 1.1412 | 1e-5 |
| X11c | 10.8489 | 1.1016 | 1e-5 |
| X9c | 11.638504532373576 | 1.9450241108176543 | 0.05 |
| X13c | 13 | 2 | 0.05 |
| X15c | 15 | 2 | 0.05 |
| X13w | 13 | 1.1313637847840483 | 0.05 |
| X15w | 15 | 1.1402972273935241 | 0.05 |
| H5c | 7.54657644033432 | 1.9923607110977173 | 1e-5 |
| H7c | 7.65995769720334 | 1.3256180959050483 | 0.05 |
| H11c | 10.125807959857049 | 1.153533260758028 | 0.05 |
| H9c | 6.932199328414673 | 0.8691859351054253 | 0.05 |
| H13c | 13 | 2 | 0.05 |
| H15c | 10.986580759286882 | 0.9586166981384515 | 0.05 |
| H13w | 10.324190393534005 | 0.967547125643583 | 0.05 |
| H15w | 10.986580759286882 | 0.8409175941229751 | 0.05 |

X5c's nominal bound is half its reference infidelity, 8.27108241444239e-5;
its final residual is 8.271082413894101e-5. Other final numerical controls
impose the full rectangular nominal target, not a nominal finite-edge target.
The smooth maximum is logsumexp(50 I/tau)/50 <= 1 over the protected samples;
it upper-bounds each sampled I/tau, not the continuum maximum.

## Noise in design and evaluation

The edge-aware Hadamard searches optimize deterministic rectangular and area-preserving finite-edge models. The latter uses 4 ns cosine ramps at a common peak Rabi frequency of 5 MHz. Gaussian detuning noise enters candidate selection and is therefore part of the design procedure.

Selection uses seed 2026102201, 96 modes, and 16 antithetic trajectories comprising eight independent Gaussian pairs. The ensemble is shared across candidates. Central selection uses rectangular 101x101 and noisy-edge 31x31 grids on B0.15; width selection uses 201x201 and 61x61 grids on B0.5. H9c uses a 401x401 rectangular mean grid. The noise spectrum is A/f with A=3.6e5 Hz^2 on 1 Hz--1 MHz.

The fixed-control domain calculation uses seed 2026102242, 96 modes, an 81x81 grid, and 128 trajectories comprising 64 independent antithetic pairs. The nominal cutoff sweep uses seed 2026102243, 192 modes, and 512 trajectories comprising 256 pairs. Both use fresh Gaussian coefficients relative to candidate selection. Cutoff points share coefficients and are correlated. Fresh realizations test the specified noise model; they do not constitute a test under a different model.

## Selection specification

For X5c, the coefficients, feasibility constraints, and coverage/nominal-error trade-off are specified, but a unique scalar final ranking or tie-break is not. For X13w, connected-area selection is specified, but an executable final selector and tie-break are not supplied. The search loss and total threshold fraction are distinct from connected coverage. These limits concern candidate selection and do not affect reproduction of the fixed controls or their reported performance.
