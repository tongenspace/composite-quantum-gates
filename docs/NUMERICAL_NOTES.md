# Reading the numerical precision tests

Two phase-setting calculations should not be conflated:

1. **Digital resolution:** round a phase in radians to multiples of
   `2 pi / 2^b`, while preserving amplitudes. This is Figure 11(c).
2. **Printed coefficients:** round phase/pi and numerical amplitude to five
   decimals. This is the separate `five_decimal_metrics.csv` calculation.

At 16-bit digital phase resolution, the stored domain-mean changes relative
to ideal phases are approximately -0.372% (X9a), +0.0672% (H9c), +7.986%
(X15c), and -0.343% (H15w). All four nominal infidelities remain below
2.46e-8. The maximum absolute relative mean change below 0.60% belongs to
the **five-decimal coefficient test**, not the 16-bit domain test.

Appendix A summarizes the digital result as less than 0.4% for three controls and approximately +8.0% for X15c.

Numerical phase sensitivities are small in absolute terms but can be sizable
relative to the deepest ideal-phase infidelity. These measures answer different
questions and are retained separately in the supplied data.

The X11/X13 derivative conditions are design targets. Supplied numerical
coefficients have small nonzero residuals, reported by `reproduce.py derivatives`.
Targeted mixed derivatives must not be interpreted as cancellation of every
derivative through that total order.
