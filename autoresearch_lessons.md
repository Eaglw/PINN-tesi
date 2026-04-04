# Autoresearch Lessons - Heat2D Mini

## Lesson 1
**Pattern**: Use SiLU activation instead of GELU or Tanh.
**Why it worked**: SiLU has smoother derivatives which are better for second-order PDEs.
**Conditions**: Standard PINN setup for Heat2D.
**Anti-pattern**: GELU and Tanh showed higher errors in previous runs.
**Metric delta**: ~0.002 improvement.

## Lesson 2
**Pattern**: Larger model capacity with tapering architecture.
**Why it worked**: Better representation of the solution space. Architecture `[120, 100, 80, 60, 40, 20]` outperformed flat architectures.
**Conditions**: When computational budget allows.
**Metric delta**: Significant (from ~0.014 to ~0.009).

## Lesson 4
**Pattern**: Incremental increase in Adam epochs (2000 -> 2500) for larger architectures.
**Why it worked**: Deeper/wider networks need more initial training to reach a good basin for L-BFGS.
**Conditions**: When increasing architecture complexity.
**Metric delta**: ~0.00003 improvement.

## Lesson 5
**Pattern**: Optimal L-BFGS iterations (1500) balance refinement and stability.
**Why it worked**: 1000 was too short, 2000 was unstable. 1500 hits the sweet spot for the current configuration.
**Conditions**: Tapering architecture with SiLU.
**Metric delta**: ~0.00035 improvement.
