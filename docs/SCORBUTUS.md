# SCORBUTUS benchmark

Primary source: Kukita, Kiya and Kondo, JPSJ 91, 104001 (2022),
https://doi.org/10.7566/JPSJ.91.104001 and https://arxiv.org/abs/2112.12945.
The construction uses the source's Eqs. (5) and (8), with unnormalized sinc
sin(u)/u on the (0, pi] root branch. See the manuscript's reference appendix
for chronological angles, negated phases and durations at common peak.

Run from the repository root:

```sh
python reproduce.py scorbutus
python reproduce.py tables --only SCORBUTUS_X SCORBUTUS_H
python scripts/plot_resources.py
```

The first command reconstructs both controls, recomputes 801x801 midpoint
statistics on B0.15 and B0.5, and additionally samples the B0.15 boundary for
the reported maximum. It checks independent direct matrix exponentials,
nominal targets up to global phase, first amplitude/detuning derivatives
and peak rescaling. Results go to outputs/scorbutus/results.json.
The second command uses the standard table engine; it never optimizes controls.
The final command redraws Figure 9 from the verified frozen metrics, emitting
two points in a separate SCORBUTUS legend family and a 41-row plotted-data CSV.

| Target | N | Normalized area | Mean B0.15 | Maximum B0.15 | f4 B0.15 (%) | Connected f4 B0.5 (%) |
|---|---:|---:|---:|---:|---:|---:|
| X | 5 | 5.0000000000 | 0.001864212933466 | 0.015244028525899 | 18.866398276 | 1.696848976 |
| Rx(pi/2) | 5 | 8.7046357845 | 0.001561622019911 | 0.014820119267139 | 24.172811451 | 2.175651222 |

The manuscript table entries agree with these values at their displayed precision. The executable controls are in `data/controls.json`, the table data in `data/static_metrics.json`, and the reconstruction in `data/scorbutus_verified.json`.

The metric is I = (4 - |Tr(G^dagger U)|^2)/6. For the source trace overlap Ftr = |Tr(G^dagger U)|/2, I = (2/3)(1 - Ftr^2). The source's sigma_y phase convention is opposite to the manuscript's, so all source phases are negated. Its detuning parameter maps as f = -delta_c/(1+epsilon), with the physical detuning held fixed independently of amplitude error.
