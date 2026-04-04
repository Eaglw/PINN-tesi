# Autoresearch Log - Heat2D Mini L2 Optimization

**Goal**: Decrease `L2_Relative_Error` for the `heat2dmini` setup.
**Metric**: `L2_Relative_Error` (lower is better).
**Scope**: `heat2dmini/Heat2D_adaptive_mini.py`, `Heat2D/src/Heat2D_PINN.py`.
**Verification Command**: `.\venv\Scripts\python.exe heat2dmini/verify_metric_fast.py`.

## Theoretical Background & Assumptions
1. **PINNs (Physics-Informed Neural Networks)**: Minimize a composite loss function consisting of:
    - **Physics Residual**: PDE residual over collocation points.
    - **Boundary Conditions (BC)**: Residuals on boundary points.
    - **Initial Conditions (IC)**: Not applicable for static Laplace/Heat 2D.
2. **Adaptive Activations**: Learnable slope `a` for the activation function (`f(x) = act(a*x)`) can help the network capture sharp gradients or high-frequency components faster.
3. **Weighting Strategies**: Dynamic weighting (Learning Rate Annealing) helps balance the competition between BC and PDE losses, which often have different gradient magnitudes.

## Iteration Log

### Iteration 0: Baseline
- **Description**: Initial run with default parameters (120x100x80x60x40x20 architecture, GELU activation, 2000 Adam + 1000 L-BFGS epochs).
- **L2 Relative Error**: 0.011182
- **Status**: Baseline.

### Iteration 1: SiLU Activation
- **Description**: Changed default activation from GELU to SiLU.
- **Assumption**: SiLU (Swish) has smoother second-order derivatives, which is beneficial for solving second-order PDEs like Laplace.
- **L2 Relative Error**: 0.008913
- **Status**: Keep (Improvement: ~20%).

