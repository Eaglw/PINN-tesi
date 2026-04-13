# Autoresearch Lessons - Heat2D Mini

## Lesson 1 — Activation Functions
**Pattern**: SiLU (Swish) activation outperforms GELU and Tanh for Heat2D.
**Why it worked**: SiLU has smoother second-order derivatives, which is beneficial for minimizing residuals of second-order PDEs like the Laplace equation.
**Metric delta**: ~0.002 improvement from GELU.

## Lesson 2 — Model Architecture
**Pattern**: Tapering architecture (e.g., `[120, 120, 100, 80, 60, 40, 20]`) is the most stable.
**Why it worked**: Higher initial capacity captures coordinate features, while tapering regularizes the flow. Expansion to 130x2...30 (Iter 71) did not improve results, suggesting 120 is the sweet spot.
**Metric delta**: Significant (from ~0.014 to ~0.007).

## Lesson 3 — Optimization Strategy
**Pattern**: 2500 Adam + 1500 L-BFGS (history_size=300).
**Why it worked**: Adam reaches the basin, L-BFGS refines. Increasing history_size to 400 or 500 (Iter 42, 65) did not help, nor did increasing epochs to 3000 (Iter 27, 67).
**Metric delta**: Stable convergence.

## Lesson 4 — Boundary Alignment
**Pattern**: Optimal `bc_weight` is exactly 25.0 with 100 points per side.
**Anti-pattern**: Increasing to 150 points (Iter 45, 70) regressed the metric, likely due to gradient imbalance.

## Lesson 5 — Sparse Internal Supervision (CRITICAL)
**Pattern**: **50 anchor points** from the analytical solution with `lambda_data=10.0`.
**Why it worked**: Provides a "compass" for the optimizer, preventing it from settling in physically stable but numerically inaccurate basins (like the 0.007091 attractor).
**Anti-pattern**: 10 points are too few (Iter 58), 100 points are too many (Iter 60).
**Metric delta**: **-0.00045** (Breakthrough to 0.00635).

## Lesson 6 — Basin Dominance
**Pattern**: L-BFGS converges to the same exact L2 error (0.006353) despite minor initialization or scheduling changes.
**Why it worked**: The combination of 50 anchor points and seed 123 defines a very deep and wide basin of attraction. Once the network enters this basin during Adam, L-BFGS consistently finds the same local minimum.
**Strategy**: To break this plateau, a fundamental change in the sampling logic (like RAR) is required to redefine the loss surface.

## Lesson 10 � Iterations 93-119: The 0.00635 Plateau
**Pattern**: Advanced architectural and optimization changes (HLConcPINN, Hybrid Act, Per-neuron Adaptivity, Importance Sampling) consistently regressed or plateaued at exactly 0.006353.
**Why it worked (Inertia)**: The SiLU-Tapered architecture with 50 anchor points appears to have found a very deep and wide basin of attraction. Perturbations like noise or LR restarts eventually lead back to this same state.
**Conditions**: High-precision Laplace 2D with specific Sobol seed (123).
**Anti-pattern**: Increasing Nx to 500 for BC targets caused massive regression (0.0418) and numerical instability without exponential stabilization.
**Metric delta**: 0.0 (Hard plateau hit).

