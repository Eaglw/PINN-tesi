# Autoresearch Lessons - Heat2D Mini

## Lesson 1 — Iterations 1-34 (Cumulative)
**Pattern**: Feature Scaling and Domain Mapping.
**Why it worked**: Scaling coordinates to [-1, 1] (Iteration 21) was a major breakthrough. It ensures that inputs to the neural network are within a well-behaved range for common activation functions like GELU, preventing saturation and improving gradient flow.
**Conditions**: Essential for all PINN problems where the physical domain is not already centered at zero.
**Anti-pattern**: Training on raw physical coordinates (e.g. [0, 1]) without centering.
**Metric delta**: -0.00826819 (BREAKTHROUGH).

## Lesson 2 — Iterations 3-13 (Cumulative)
**Pattern**: Quasi-random Collocation Sampling.
**Why it worked**: Switching from random to Sobol sampling (Iteration 3) provided more uniform domain coverage with fewer points, leading to more robust physics residual enforcement and better generalization.
**Conditions**: Effective when the number of collocation points is limited.
**Anti-pattern**: Pure random sampling or very sparse grids.
**Metric delta**: -0.00798106.

## Lesson 3 — Iterations 33-34
**Pattern**: Adaptive Loss Weighting (Balancing).
**Why it worked**: Reducing the initial `bc_weight` from 50.0 to 20.0 (Iteration 34) allowed the Adam phase to better balance the boundary enforcement with the interior physics. Too high BC weight makes the optimization problem "stiff" and prevents the interior from being resolved correctly.
**Conditions**: Use in conjunction with Dynamic Weighting (Learning Rate Annealing).
**Anti-pattern**: Extreme BC weights (>100 or <10) at the start of training.
**Metric delta**: -0.00134182.

## Lesson 4 — Iterations 1-26 (Cumulative)
**Pattern**: Architecture Tapering.
**Why it worked**: Deep tapered architectures (e.g. [120, 100, 80, 60, 40, 20]) provide a good balance between representational capacity in early layers and regularization/focus in later layers. Uniform or shallow/wide architectures were consistently worse.
**Conditions**: Best for problems with smooth solutions like the Laplace equation.
**Anti-pattern**: Uniform width architectures (e.g. 6x100) or very shallow networks.
**Metric delta**: -0.00955558.
