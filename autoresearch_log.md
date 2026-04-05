# Autoresearch Log - Heat2D Mini L2 Optimization

**Goal**: Minimize `L2_Relative_Error` for the `heat2dmini` setup.
**Metric**: `L2_Relative_Error` (lower is better).
**Scope**: `heat2dmini/Heat2D_adaptive_mini.py`, `Heat2D/src/Heat2D_PINN.py`.
**Verification Command**: `.\venv\Scripts\python.exe heat2dmini/verify_metric_fast.py`.

## Theoretical Background
The project uses Physics-Informed Neural Networks (PINNs) to solve the 2D Heat equation (Laplace) on a square domain. The solution is compared against an analytical series expansion. Key components include:
- **Adaptive Activations**: Learnable slope parameters for each layer to improve gradient capture.
- **Dynamic Weighting**: Learning Rate Annealing to balance boundary and physics losses.
- **Tapering Architecture**: A strategy to optimize network capacity versus training stability.

## Research History Summary

### Phase 1: Foundation (Iter 0-10)
Discovery that SiLU and tapering architectures are significantly better than flat ones with Tanh or GELU.

### Phase 2: Capacity Expansion (Iter 11-20)
Introduction of extra wide layers and increased training epochs (2500 Adam). Reached L2 < 0.008.

### Phase 3: Precision Refinement (Iter 21-33)
Optimization of L-BFGS iterations (settled on 1500) and investigation of adaptive activation initializations. Reached the current best of **0.00680**.

### Phase 4: Consolidation (Current)
Consolidated results from various independent test series into a unified workflow to avoid redundant experiments.

## Current Best Configuration (Iter 24)
- **Architecture**: `ADAPTIVE_[120, 120, 100, 80, 60, 40, 20]`
- **Activation**: SiLU (Adaptive)
- **Optimization**: 2500 Adam + 1500 L-BFGS
- **BC Weight**: 25.0
- **Sampling**: Sobol (1600 points, 0.02 margin)
