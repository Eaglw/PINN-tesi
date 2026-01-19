# Track Specification: Grid-based NN Training vs Random Sampling

## Overview
This track aims to implement a new training mode, `Heat2D_NN_griglia.py`, which trains a standard supervised Neural Network (NN) for the 2D Heat Transfer problem using a deterministic grid of points instead of random sampling. This functionality will be integrated into the existing `Heat2D_main.py` workflow, allowing users to select this mode via the `goal` configuration. Furthermore, the main script will be enhanced to automatically compare the two approaches (Random vs. Grid) when both are executed.

## Functional Requirements
- **New Module:** Create `Heat2D/Heat2D_NN_griglia.py` that implements supervised training using `torch.meshgrid` for data generation, matching the logic in `Heat2D_PINN.py`.
- **Main Integration:** Update `Heat2D/Heat2D_main.py` to support a new `goal` index (e.g., `5`) that triggers the grid-based NN training.
- **Automated Comparison:** Modify `Heat2D/Heat2D_main.py` to detect when both NN (Random, `goal=0`) and NN (Grid, `goal=5`) are selected. In this case, perform a post-training comparison.
- **Visualization:**
    - Generate an overlay plot of Loss curves (Training & Validation) for both approaches.
    - Produce point-wise error maps (absolute difference vs. analytical solution) for both models.

## Non-Functional Requirements
- **Consistency:** Ensure training hyperparameters (epochs, learning rate, architecture) are identical between the two NN modes.
- **Reproducibility:** Use fixed seeds to ensure that the "Random" sampling is reproducible for consistent comparisons.

## Acceptance Criteria
- `Heat2D/Heat2D_NN_griglia.py` is created and functional.
- Running `Heat2D_main.py` with the new goal triggers the grid-based training.
- Running `Heat2D_main.py` with both goals (Random + Grid) produces:
    - Individual training results.
    - A comparison loss plot.
    - Error map visualizations for both.

## Out of Scope
- Refactoring `Heat2D_PINN.py` or other modules not directly involved in this comparison.
