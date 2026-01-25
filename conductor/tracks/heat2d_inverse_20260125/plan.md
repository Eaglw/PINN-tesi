# Implementation Plan - Heat2D Inverse Problem

This plan outlines the steps to implement the inverse problem solver for the 2D Heat Transfer equation, including data generation, parameter estimation, logging, and visualization.

## Phase 1: Infrastructure and Data Generation
Setup the necessary directories and utility functions to generate synthetic, noisy observation data.

- [x] Task: Create output directories `Heat2D/experiments_inverse/` and initialize `Heat2D/results_inverse.csv`.
- [x] Task: Implement a data generation utility in `Heat2D/src/physics.py` or within the new script that samples exact solution points and adds Gaussian noise.
- [~] Task: Conductor - User Manual Verification 'Phase 1: Infrastructure and Data Generation' (Protocol in workflow.md)

## Phase 2: Core Inverse Solver Implementation
Implement the `Heat2D_inverse_main.py` script with the learnable parameter $\alpha$.

- [ ] Task: Define the `Heat2D_PINN_Inverse` model (or adapt existing) where `alpha` is an `nn.Parameter`.
- [ ] Task: Implement the training loop supporting separate learning rates for network weights and the `alpha` parameter.
- [ ] Task: Implement the physics loss calculation using the learnable `alpha`.
- [ ] Task: Implement the logic to track the history of the `alpha` parameter during training.
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Core Inverse Solver Implementation' (Protocol in workflow.md)

## Phase 3: Logging, Visualization, and Grid Search
Integrate the logging system and implement the grid search for hyperparameters.

- [ ] Task: Implement the specialized plotting function for parameter convergence (Value vs. Epoch).
- [ ] Task: Update the CSV logging to include the relative error of the estimated $\alpha$.
- [ ] Task: Implement the grid search loop to iterate over architecture, noise levels, and data density.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Logging, Visualization, and Grid Search' (Protocol in workflow.md)

## Phase 4: Verification and Finalization
Verify the inverse solver performance and ensure all artifacts are correctly generated.

- [ ] Task: Run a baseline inverse experiment with zero noise to confirm parameter recovery.
- [ ] Task: Run a full grid search and verify the results in `results_inverse.csv`.
- [ ] Task: Conductor - User Manual Verification 'Phase 4: Verification and Finalization' (Protocol in workflow.md)
