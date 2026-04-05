# Autoresearch Lessons - Heat2D Mini

## Lesson 1 — Activation Functions
**Pattern**: SiLU (Swish) activation outperforms GELU and Tanh for Heat2D.
**Why it worked**: SiLU has smoother second-order derivatives, which is beneficial for minimizing residuals of second-order PDEs like the Laplace equation.
**Conditions**: Standard PINN setup.
**Anti-pattern**: Tanh often leads to higher errors (0.14 vs 0.008) in pure physics setups.
**Metric delta**: ~0.002 improvement from GELU.

## Lesson 2 — Model Architecture
**Pattern**: Tapering architecture (e.g., `[120, 100, 80, 60, 40, 20]`) is more effective than flat architectures.
**Why it worked**: Higher initial capacity allows better mapping of input coordinates, while tapering reduces parameters in deeper layers, aiding regularization. Adding an extra wide layer (Iter 13) further improved accuracy.
**Conditions**: When computational budget allows.
**Metric delta**: Significant (from ~0.014 to ~0.007).

## Lesson 3 — Optimization Strategy
**Pattern**: Incremental increase in Adam epochs (2000 -> 2500) and L-BFGS iterations (1000 -> 1500).
**Why it worked**: Deeper/wider networks need more initial training (Adam) to reach a stable basin. L-BFGS then provides high-precision refinement.
**Conditions**: When increasing architecture complexity.
**Metric delta**: ~0.0004 improvement cumulative.

## Lesson 4 — Boundary Alignment
**Pattern**: Optimal `bc_weight` is around 22.0 - 25.0.
**Why it worked**: Balances the sharp gradients at the boundary with the smooth physics residual in the interior.
**Conditions**: 2D Heat problem with Dirichlet/Neumann conditions.
**Anti-pattern**: Weights > 30.0 or < 15.0 often lead to worse L2 errors.
**Metric delta**: ~0.003 improvement.

## Lesson 5 — Sampling & Collocation
**Pattern**: Sobol points with a 0.02 margin are standard; Halton points showed no significant gain.
**Why it worked**: Sobol provides better space-filling properties than random sampling. A small margin prevents singularities at boundaries while enforcing physics close enough to them.
**Anti-pattern**: Reducing margin to 0.01 (Iter 29) or increasing density to 50x50 (Iter 32) regressed the metric.
