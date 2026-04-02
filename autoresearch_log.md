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

## Iteration 32
- **Hypothesis**: Reduce margin to 0.01 to include more collocation points near boundaries.
- **Metric**: 0.01113035 (regressed).
- **Status**: Discarded.
- **Observation**: Including more points very close to the boundary might have destabilized the physics loss or increased competition with BC loss.
## Iteration 71
- **Hypothesis**: Adjust ReduceLROnPlateau factor to 0.4.
- **Metric**: 0.00898244 (no change).
- **Status**: Discarded.
- **Observation**: Factor change didn't trigger any improvement, suggesting LR drops are not the bottleneck.

---

## Iteration 72
- **Hypothesis**: Reduce ReduceLROnPlateau patience (600 -> 200) and cooldown (3000 -> 400) to allow LR steps within 2000 Adam epochs.
- **Status**: Discarded.
- **Metric**: 0.00905747 (regressed).
- **Observation**: Increased scheduler sensitivity led to premature LR decay, hurting Adam phase exploration.

---

## Iteration 73
- **Hypothesis**: Increase adaptive activation initialization 'a' to 1.5.
- **Status**: Discarded.
- **Metric**: 0.00992659 (regressed).
- **Observation**: Steeper initial activations likely caused gradient instability or poor conditioning early in training.

---

## Iteration 74
- **Hypothesis**: Set adaptive activation initialization 'a' to 1.0 (standard profile).
- **Status**: Discarded.
- **Metric**: 0.01256952 (regressed).
- **Observation**: Standard GELU profile (a=1.0) is significantly less effective than the current best (a=1.1) for this specific problem.

---

## Iteration 75
- **Hypothesis**: Increase dynamic weight update frequency (100 -> 50).
- **Status**: Discarded.
- **Metric**: 0.00917206 (regressed).
- **Observation**: More frequent updates might have introduced instability in the gradient balancing EMA, preventing Adam from reaching a better minimum.

---

## Iteration 76
- **Hypothesis**: Increase collocation points (40x40 -> 50x50).
- **Status**: Discarded.
- **Metric**: 0.00983501 (regressed).
- **Observation**: Higher resolution collocation might require more Adam epochs or higher model capacity to be beneficial.

---

## Iteration 77
- **Hypothesis**: Refine BC weight to 18.0 (slight reduction from 20.0).
- **Status**: Discarded.
- **Metric**: 0.00955015 (regressed).
- **Observation**: bc_weight=20.0 remains the local optimum for balancing BC and Interior physics losses.

---

## Iteration 78
- **Hypothesis**: Reduce Adam Learning Rate (1e-3 -> 5e-4).
- **Theoretical Assumption**: A smaller learning rate provides more stable updates, potentially reaching a better local minimum before L-BFGS takes over.
