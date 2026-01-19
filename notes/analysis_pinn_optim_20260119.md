# Analysis Report: PINN Performance & Optimization (2026-01-19)

## Phase 1: Comparative Analysis

### 1. Baseline Comparison (Grid NN vs PINN)
- **Objective**: Compare the error distribution of a standard NN trained on grid data versus the PINN trained with Physics Loss.
- **Execution**: Run `Heat2D/Heat2D_compare_PINN_vs_Grid.py`.
- **Results**:
    - Comparison Error Map saved to: `Heat2D/Results/comparison_pinn_grid/Comparison_ErrorMap_Grid_vs_PINN.png`.
    - Loss histories saved in the same directory.
    - **Observation**: The PINN typically shows higher relative error peaks compared to the Grid NN, likely due to optimization difficulties in balancing the competing loss terms (Data vs PDE vs BC).

### 2. Gradient Analysis
- **Objective**: Investigate if "Gradient Pathologies" (imbalanced gradients) are hindering convergence.
- **Execution**: Run `Heat2D/Heat2D_gradient_analysis.py` (5000 epochs).
- **Methodology**: Logged the L2 norms of the gradients for `data_loss`, `bc_loss`, and `pde_loss` every 10 epochs.
- **Results**:
    - Gradient Norm Plot saved to: `Heat2D/Results/gradient_analysis/PINN_gradients.png`.
    - **Hypothesis**: If one gradient term is orders of magnitude smaller than others, it may be ignored by the optimizer. Typically, PDE gradients can be smaller or noisier than Data gradients initially.

## Next Steps (Phase 2)
Based on these findings (and the general literature on PINNs), we will proceed with:
1.  **Loss Balancing**: Implementing weights to balance the terms.
2.  **Collocation Improvement**: Increasing physics resolution.
