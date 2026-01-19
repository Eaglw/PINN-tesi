# Optimization Report: PINN Heat2D (2026-01-19)

## Overview
This track focused on analyzing and optimizing the performance of a Physics-Informed Neural Network (PINN) for the 2D Heat Equation. The initial baseline showed high relative error compared to a standard Neural Network trained on grid data.

## Key Findings

### 1. Comparative Analysis (Phase 1)
- **Baseline Performance:** The standard PINN struggled to match the accuracy of a purely data-driven NN (Grid), exhibiting "Gradient Pathologies" where data loss gradients likely dominated or conflicted with physics gradients initially.
- **Gradient Analysis:** Logging revealed significant differences in gradient magnitudes, but simple re-weighting was not the immediate solution.

### 2. Optimization Experiments (Phase 2)
- **Experiment A: Loss Balancing (Negative Result)**
    - *Hypothesis:* Increasing physics weight ($\lambda_{pde} = 0.5$) would force better physical compliance.
    - *Result:* **Degraded performance**. The higher stiffness of the physics constraint likely trapped the optimizer in local minima early on.
    - *Conclusion:* Stick to lower physics weights (e.g., $\lambda_{pde} = 0.05$) to allow data-driven "warmup" of the solution shape.

- **Experiment B: Collocation Density (Positive Result)**
    - *Hypothesis:* Increasing collocation points from 50x50 (2.5k) to 100x100 (10k) would reduce aliasing and improve integral approximation.
    - *Result:* **Significant Improvement**. Final loss dropped by ~50% in short runs.
    - *Conclusion:* Higher resolution is critical for resolving the PDE constraints effectively.

### 3. Final Verification (Phase 3)
- **Unified Test:** Compared Baseline (50x50, 30k epochs) vs Optimized (100x100, 30k epochs).
- **Final Results:**
    - **Baseline Loss:** ~2.00e-03
    - **Optimized Loss:** **8.86e-04** (Over 2x improvement)
    - **Visual Inspection:** The error maps show a substantial reduction in peak relative error across the domain.

## Recommendations for Future Tracks
- **Default Configuration:** Adopt `n_collocation=100` (or higher) as the standard for future 2D problems.
- **Training Strategy:** Maintain the "low physics weight" approach or investigate dynamic weighting (e.g., annealing) rather than static boosting.
- **Next Steps:** Consider investigating "Hard Constraint" architectures or different activation functions (SIREN) if further precision is needed ($<1e-4$).
