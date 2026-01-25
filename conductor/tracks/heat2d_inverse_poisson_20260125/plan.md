# Implementation Plan - Heat2D Inverse Problem

This plan outlines the steps to implement the inverse problem solver for the 2D Heat Transfer (Poisson) equation, ensuring consistency with the direct problem's geometry and BCs.

## Phase 1: Infrastructure and Ground Truth Generation [checkpoint: 6057283]
Setup directories and implement the synthetic data generation for the Poisson problem.

- [x] Task: Create output directory `Heat2D/experiments_inverse/` and initialize `Heat2D/results_inverse.csv` with optimized column ordering.
- [x] Task: Implement the "Poisson Ground Truth" generator. This will solve the forward problem (T=1 at x=1, else T=0, with Q=1) using a high-precision PINN or numerical method to create a dataset for the inverse solver.
- [x] Task: Conductor - User Manual Verification 'Phase 1: Infrastructure and Ground Truth Generation' (Protocol in workflow.md)

## Phase 2: Core Inverse Solver Implementation [checkpoint: bd99fdd]
Implement the `Heat2D_inverse_main.py` script with the learnable parameter $k$.

- [x] Task: Implement the `InversePoissonPhysics` class in `Heat2D/src/inverse_physics.py`. It should calculate the residual $k(T_{xx} + T_{yy}) + 1 = 0$.
- [~] Task: Implement the training loop in `Heat2D_inverse_main.py` that optimizes both model weights and the $k$ parameter.
- [~] Task: Implement a parameter tracking system to record the evolution of $k$ during training.
- [x] Task: Conductor - User Manual Verification 'Phase 2: Core Inverse Solver Implementation' (Protocol in workflow.md)

## Phase 3: Logging, Visualization, and Grid Search [checkpoint: 1a07c1f]
Integrate specialized plotting and implement the grid search loop.

- [x] Task: Implement specialized visualization: side-by-side comparison of temperature fields and the $k$ convergence plot (Value vs Epoch).
- [x] Task: Implement the grid search logic, iterating over noise levels, data density, and network architecture.
- [x] Task: Ensure CSV logging correctly calculates the relative error for the estimated $k$.
- [x] Task: Conductor - User Manual Verification 'Phase 3: Logging, Visualization, and Grid Search' (Protocol in workflow.md)

## Phase 4: Verification and Finalization [checkpoint: 660889b]
Validate the solver's performance.

- [x] Task: Run a baseline test with zero noise to verify $k$ recovery within the specified tolerance (<2%).
- [x] Task: Execute the full grid search and verify the consistency of `results_inverse.csv`.
- [x] Task: Conductor - User Manual Verification 'Phase 4: Verification and Finalization' (Protocol in workflow.md)
