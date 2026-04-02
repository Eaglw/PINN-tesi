# Autoresearch Log - Heat2D Mini L2 Error Optimization

## Objective
Diminuire l'L2 Error del setup heat2dmini.
Metric: L2_Relative_Error (lower is better).

## Baseline
- **Iteration 0**: 0.01032426
- **Date**: 2026-04-02
- **Command**: `.\venv\Scripts\python.exe heat2dmini/verify_metric_fast.py`

## Theoretical Assumptions
1. **L-BFGS Convergence**: PINNs often benefit from second-order optimization (L-BFGS) after an initial Adam phase to reach high precision. 500 iterations might be insufficient for full convergence.
2. **Adaptive Activations**: The learnable parameter 'a' in `AdaptiveActivation` allows the network to adapt the slope of the activation function, potentially mitigating vanishing/exploding gradients and improving expressivity.
3. **Coordinate Scaling**: Mapping coordinates to [-1, 1] is standard practice in PINNs to improve training stability and ensure features are on the same scale.
4. **Sampling Strategy**: Quasi-random sequences like Sobol provide better domain coverage than pure random sampling, leading to more robust physics residuals.

---

## Iteration 31
- **Hypothesis**: Increasing L-BFGS iterations from 500 to 2000.
- **Metric**: 0.01032426 (no change).
- **Status**: Discarded.
- **Observation**: Converge tolerance was likely reached before 500 iterations.
