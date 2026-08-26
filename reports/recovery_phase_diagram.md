# Finite-sample chord-lobe recovery

The phase experiment contains 44,000 independent Gaussian-noise draws: four
fixed lobe configurations, eleven normalized curvature strengths, and 1000
replications per cell.

The sufficient theorem boundary is `rho = gamma/tau > 2`. Across all eight
cells above that boundary, exact first-level knot recovery was 1.000. The
smallest cellwise 95% Wilson lower bound was 0.9962. The joint event comprising
exact knots and both stated reconstruction bounds had minimum probability
0.9980 and minimum Wilson lower bound 0.9927.

The empirical transition was concentrated below the conservative boundary:
exact recovery ranged from 0.433 to 0.879 at `rho=1.45`, from 0.980 to 0.991 at
`rho=1.75`, and was 0.999 in every configuration at `rho=1.90`.

Reproduce with:

```bash
python experiments/run_recovery_phase_diagram.py
python experiments/generate_recovery_phase_figure.py
```

Machine-readable trial, aggregate, metadata, and summary files are in
`results/recovery_phase_r1/`.
