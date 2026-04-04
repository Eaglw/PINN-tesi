# Autoresearch Log - Heat2D Mini L2 Optimization (Session 2026-04-04)

**Goal**: Decrease `L2_Relative_Error` for the `heat2dmini` setup.
**Metric**: `L2_Relative_Error` (lower is better).
**Scope**: `heat2dmini/Heat2D_adaptive_mini.py`, `Heat2D/src/Heat2D_PINN.py`.
**Verification Command**: `.\venv\Scripts\python.exe heat2dmini/verify_metric_fast.py`.

## Theoretical Background & Assumptions
1. **PINNs (Physics-Informed Neural Networks)**: Minimize PDE residual and BC residuals.
2. **Adaptive Activations**: Learnable slope `a` helps capture sharp gradients.
3. **Architecture**: Model capacity and depth significantly affect convergence and precision.

## Iteration Log

### Iteration 0: Baseline
- **Description**: Default parameters (80x80x80x80 architecture, SiLU activation, 2000 Adam + 1000 L-BFGS).
- **L2 Relative Error**: 0.014140
- **Status**: Baseline.
