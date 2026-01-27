# Specification - Heat2D Loss Weighting Implementation

## Overview
This track aims to improve PINN convergence and accuracy for the 2D Heat Transfer problem by introducing a static weighting mechanism for the loss components (Physics, Boundary, and Data). Analysis of previous results (n_points=2000) showed that Boundary Loss often dominates other components by an order of magnitude, potentially leading to sub-optimal training.

## Functional Requirements
- **Weighting Strategy:** Implement static weights for loss components. Based on initial analysis, the target weights are:
    - $\lambda_{bc} = 1$ (Boundary Condition)
    - $\lambda_{phys} = 10$ (Physics Residual)
    - $\lambda_{data} = 50$ (Data Loss)
- **New Script:** Create `Heat2D/Heat2D_weighted_main.py` based on `Heat2D_main.py`.
- **Output Organization:** 
    - Save experiment artifacts in `Heat2D/experiments/weighted/`.
    - Update the main `Heat2D/results.csv` to include a `Loss_Weight` column.
    - Existing entries in `results.csv` must be labeled as `not_weighted`.
    - New weighted entries must be labeled with a descriptive string like `BC=1-PHYS=10-DATA=50`.
- **Grid Search:** Execute a grid search over the following hyperparameters:
    - **Activation Functions:** Tanh, SiLU, GELU.
    - **Epochs:** 20000, 40000.
    - **Learning Rate Scheduler:** Fixed LR, Step Decay.
    - **Run Types:** `PINN_DataPhys` (with data), `PINN_PurePhys` (purely physical).
    - **Fixed Architecture:** 4 hidden layers of 50 neurons (`[2, 50, 50, 50, 50, 1]`).

## Non-Functional Requirements
- **Reproducibility:** Maintain seed consistency (default 123).
- **Precision:** Ensure `torch.float64` is used.
- **Logging:** Maintain the existing logging structure for CSV and plots.

## Acceptance Criteria
- [ ] `Heat2D/Heat2D_weighted_main.py` exists and correctly implements the weighting logic.
- [ ] `results.csv` contains the `Loss_Weight` column with correct values for both old and new runs.
- [ ] Experiments are correctly saved in the `experiments/weighted` subdirectory.
- [ ] Comparison plots are generated for each run in the grid search.

## Out of Scope
- Dynamic weighting (e.g., Adaptive Weighting or Neural Tangent Kernel based methods).
- Modifying other physical problems (CSTR, etc.).
- Testing architectures other than 4x50.
