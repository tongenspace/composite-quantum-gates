# Model, units and numerical observables

## Pulse propagation

A sequence is a list of phases `phases`, amplitudes `amplitudes`, and durations
`durations_T`, in **chronological** order. The implementation is `U_N ... U_1`.
Each pulse uses the Hermitian Hamiltonian written in the README, with the
minus-y phase convention. `theta` is pi for X and pi/2 for the Hadamard primitive.
Global phases do not affect the reported gate fidelity.

`src/rectangular.py` implements both a quaternion product and direct SciPy
matrix exponentials. The quaternion formula evaluates small infidelities
without subtracting two nearly equal trace expressions. The independent
matrix calculation is used at points where that subtraction is well resolved.

## Error coordinates

- `epsilon` is a common fractional amplitude error.
- `delta = Delta/Omega_0` uses a fixed construction reference.
- `delta_c = Delta/Omega_max` uses the common physical peak.
- `p = max(a_k)`, so `delta = p delta_c`.

Do not multiply detuning by `1+epsilon`, by an individual segment's amplitude,
or by its envelope. A common duration error eta is represented exactly in the
rectangular model by `epsilon_eff=(1+epsilon)(1+eta)-1` and
`delta_eff=delta(1+eta)`.

## Gate infidelity

For a qubit, the average gate infidelity between U and G is

\[
\mathcal I(U,G)=\frac{4-|\mathrm{Tr}(G^\dagger U)|^2}{6}.
\]

The error-domain mean is a **discrete sum** over equally weighted samples.
It is a separate average from the state average already built into gate
fidelity. Averaging over Gaussian phase offsets or noise trajectories adds
another, explicitly identified ensemble average.

## Reported resource and coverage quantities

The pulse area is `Phi = pi sum(a_k durations_T[k])`. Its normalized cost is
`R_A = Phi/theta`, so an ideal primitive rotation costs one. Segment count is
reported separately. Equal area does not imply equal physical duration for
variable-amplitude sequences. The common-peak rectangular duration is
`(pi p/Omega_max) sum(durations_T)`.

The table domain B_0.15 is `abs(epsilon),abs(delta_c)<=0.15`. Means and
fractions use an 801x801 midpoint grid; maxima additionally check boundary
samples. A sampled maximum is not a certified supremum over the continuum.

`f_4(0.15)` counts all sampled points below 1e-4 infidelity in B_0.15.
The wider metric counts the connected component containing the nominal point
`(epsilon,delta_c)=(0,0)` inside B_0.5. The connected region uses four-neighbor
connectivity on the evaluation grid; separate high-fidelity islands are excluded.

[Evaluation-grid convergence](../studies/grid_convergence/README.md) is checked for the fixed X5c, H5c, X15c, and H15c controls on midpoint grids from 8x8 to 3201x3201. The 801x801 central means differ from the 3201x3201 reference by less than 0.0032%. This deterministic check changes only the evaluation mesh; it uses the same common-peak domain as the tables and does not optimize the coefficients. Sobol evaluations used in candidate selection remain specified separately in the design methods.

## Finite edges and frequency noise

Appendix B uses a common peak of 5 MHz. For each control,
`T_s=100 p ns`, and the cosine envelope preserves the rectangular area while
adding an edge time to each segment's duration. At resonance, area preservation
preserves the single-pulse unitary. Away from resonance, the drive and detuning
terms do not commute, so finite-domain performance can improve or deteriorate.

The one-sided frequency-noise PSD is `A/f`, with `A=3.6e5 Hz^2`. The principal
band is 1 Hz--1 MHz. Its RMS is `sqrt(A log(fmax/fmin))` in Hz. Realizations
are stationary Gaussian and continuous across the full gate. The mode weights
and independent Gaussian sine/cosine coefficients appear in `shaped_model.py`.
For uncertainty estimates, antithetic trajectories form independent pairs;
standard errors are computed from pair means.
