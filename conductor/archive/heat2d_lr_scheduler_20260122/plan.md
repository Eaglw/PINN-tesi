# Implementation Plan: Step Decay LR Scheduler for Heat2D

Implement a Step Decay learning rate scheduler for Heat2D experiments, integrated into the grid search and results logging.

## Phase 1: Core Logic Update (Training Scripts) [checkpoint: 38fe127]
Update individual training scripts to support the new learning rate strategy.

- [x] Task: Modify `Heat2D/src/Heat2D_NN.py` to accept `lr_strategy` and implement `StepLR`.
- [x] Task: Modify `Heat2D/src/Heat2D_NN_griglia.py` to accept `lr_strategy` and implement `StepLR`.
- [x] Task: Modify `Heat2D/src/Heat2D_PINN.py` to accept `lr_strategy` and implement `StepLR`.
- [x] Task: Conductor - User Manual Verification 'Core Logic Update' (Protocol in workflow.md)

## Phase 2: Grid Search and Logging (Main Script) [checkpoint: d0cee9d]
Update the main execution script to handle the new parameter and format the output.

- [x] Task: Update `Heat2D/Heat2D_main.py` to include `lr_strategies = ['fixed', 'step_decay']` in the grid search loop.
- [x] Task: Implement LR range calculation logic in `Heat2D/Heat2D_main.py` for logging.
- [x] Task: Update the CSV writing logic in `Heat2D/Heat2D_main.py` to handle the new LR format.
- [x] Task: Conductor - User Manual Verification 'Grid Search and Logging' (Protocol in workflow.md)

## Phase 3: Verification and Cleanup [checkpoint: 84c50f8]
Run a small test grid to verify the implementation.

- [x] Task: Run a mini-grid experiment (e.g., 100 epochs) to verify both `fixed` and `step_decay` work.
- [x] Task: Verify `results.csv` content for correct LR formatting.
- [x] Task: Conductor - User Manual Verification 'Verification and Cleanup' (Protocol in workflow.md)
