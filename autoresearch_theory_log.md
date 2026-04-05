# Autoresearch Theory Log - Heat2D Mini (Continued)

## Status as of Iteration 28
- **Best L2**: 0.0068029 (Iter 24)
- **Current Architecture**: ADAPTIVE_[2, 120, 120, 120, 100, 100, 80, 60, 40, 20, 1]
- **Current Config**: 2500 Adam epochs, 1500 L-BFGS iterations, bc_weight=25.0, adaptive act init=1.1.

## Iteration 29: Reducing Collocation Margin
**Hypothesis**: Reducing the exclusion margin for internal points from 0.02 to 0.01 will allow the PDE residual to be enforced closer to the boundaries, smoothing the transition between BC and physics.
**Change**: `margin = 0.01` in `heat2dmini/Heat2D_adaptive_mini.py`.
