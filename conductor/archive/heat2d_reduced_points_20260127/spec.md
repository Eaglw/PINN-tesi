# Specification: Reduced Points Grid Search (Heat2D Direct)

## Overview
Expand the Grid Search for the 2D Heat Transfer (Laplace) problem using purely supervised Neural Networks (NN) with a reduced dataset of ~500 points (compared to the standard ~2000). This will help evaluate how model performance degrades or holds up with limited data.

## Functional Requirements
1.  **Reduced Dataset Generation**:
    -   Internal Points: 300 total (one set for Random, one for Uniform Grid).
    -   Boundary Points: 50 per side (Total 200).
    -   Total Dataset Size: ~500 points.
2.  **Grid Search Scope**:
    -   **Architecture**: Fixed to `50x4` (4 hidden layers, 50 neurons each). The larger `80x6` architecture is excluded.
    -   **Hyperparameters**: Test all combinations of Epochs (20000, 40000), Activation Functions (Tanh, SiLU, GELU), and LR Schedulers (Fixed, Step Decay).
    -   **Methods**: Run both `NN_Random` and `NN_Grid` cases.
3.  **Data Persistence**:
    -   Save all experiment artifacts (models, plots, logs) in a dedicated folder: `Heat2D/experiments_reduced_points/`.
4.  **Results Tracking**:
    -   Update `Heat2D/results.csv` to include an `n_points` column.
    -   Perform a migration on the existing `results.csv` to set `n_points` to `2000` for all previous entries.
    -   Ensure the logging system automatically records the point count for new experiments.

## Technical Requirements
-   Modify `func/logging_utils.py` to support the `n_points` field in `update_results_csv`.
-   Create a specialized runner script `Heat2D/Heat2D_reduced_main.py` to orchestrate this specific grid search without interfering with the main script's defaults.
-   Update `Heat2D/src/Heat2D_NN.py` and `Heat2D/src/Heat2D_NN_griglia.py` if necessary to ensure they accept and log the dataset size correctly.

## Acceptance Criteria
-   `results.csv` contains the `n_points` column with `2000` for old runs and `~500` for new runs.
-   A full set of results for the `50x4` architecture is available in `Heat2D/experiments_reduced_points/`.
-   The training runs successfully for both Random and Grid point distributions.
