# Implementation Plan - Heat2D Loss Weighting

This plan outlines the steps to introduce static loss weighting in the 2D Heat Transfer PINN experiments, including data migration, script updates, and a new grid search.

## Phase 1: Infrastructure and Data Migration [checkpoint: b9bbb35]
- [x] Task: Update `Heat2D/results.csv` schema to include the `Loss_Weight` column and label all existing entries as `not_weighted`.
- [x] Task: Audit and update existing scripts (e.g., `Heat2D/Heat2D_main.py`, `Heat2D/Heat2D_reduced_main.py`, etc.) to ensure they support the new `Loss_Weight` column when logging.
- [x] Task: Create the `Heat2D/experiments_weighted/` directory.
- [x] Task: Conductor - User Manual Verification 'Infrastructure and Data Migration' (Protocol in workflow.md)

## Phase 2: Core Implementation [checkpoint: 0ee210a]
- [x] Task: Create `Heat2D/Heat2D_weighted_main.py` by adapting `Heat2D_main.py`.
- [x] Task: Implement the loss weighting logic in the training loop using $\lambda_{bc}=1, \lambda_{phys}=10, \lambda_{data}=50$.
- [x] Task: Configure the grid search in the new script (Architecture: 4x50; Activations: Tanh, SiLU, GELU; Epochs: 20k, 40k; LR: Fixed, Step Decay; Types: DataPhys, PurePhys).
- [x] Task: Update the CSV logging logic in `Heat2D/Heat2D_weighted_main.py` to record the weight string `BC=1-PHYS=10-DATA=50`.
- [x] Task: Conductor - User Manual Verification 'Core Implementation' (Protocol in workflow.md)

## Phase 3: Grid Search Execution and Validation [checkpoint: dac8ccd]
- [x] Task: Execute the comprehensive grid search using `Heat2D/Heat2D_weighted_main.py`.
- [x] Task: Verify that `results.csv` is correctly updated and that all artifacts are saved in `Heat2D/experiments_weighted/`.
- [x] Task: Perform a brief manual comparison of the weighted results against the non-weighted ones for the 2000 points case.
- [x] Task: Conductor - User Manual Verification 'Grid Search Execution and Validation' (Protocol in workflow.md)
