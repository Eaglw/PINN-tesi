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

## Lesson 3
**Pattern**: Increase L-BFGS `history_size` to 200.
**Why it worked**: Better Hessian approximation for final refinement.
**Conditions**: Final L-BFGS phase.
**Metric delta**: ~0.0003 improvement.
