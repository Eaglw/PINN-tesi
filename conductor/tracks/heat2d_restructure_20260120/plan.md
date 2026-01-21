# Implementation Plan: Heat2D Module Restructuring

Reorganize the `Heat2D` directory to separate source code from experiment artifacts, using a goal-categorized `experiments/` directory.

## Phase 1: Directory Setup and Code Migration
Establish the new folder structure and relocate core logic scripts.

- [x] Task: Create Heat2D/src/ and Heat2D/experiments/ directories.
- [x] Task: Move training scripts (Heat2D_NN.py, Heat2D_PINN.py, Heat2D_NN_griglia.py, Heat2D_optim_collocation.py, etc.) and physics.py into Heat2D/src/.
- [x] Task: Move remaining standalone scripts (comparison, verification, gradient) to Heat2D/src/.
- [x] Task: Update Heat2D/__init__.py if necessary to maintain package integrity.
- [x] Task: Conductor - User Manual Verification 'Phase 1: Migration' (Protocol in workflow.md) [checkpoint: a6c96c9]

## Phase 2: Refactor Orchestration Logic
Update `Heat2D_main.py` and the training functions to handle new paths and implement script snapshotting.

- [x] Task: Update imports in `Heat2D/Heat2D_main.py` to reference the new `src/` location.
- [x] Task: Refactor `Heat2D_main.py` to define output paths within `experiments/<Goal_Name>/`.
- [x] Task: Implement a utility function to copy the training script to the experiment folder and inject the required explanatory header.
- [x] Task: Update `train_model` functions (NN, PINN, etc.) to accept the new `experiments` path structure for saving plots and GIFs.
- [x] Task: Conductor - User Manual Verification 'Phase 2: Orchestration' (Protocol in workflow.md) [checkpoint: 5340689]

## Phase 3: Migration of Existing Results
Organize current artifacts from `Results/` into the new `experiments/` structure with documentation.

- [x] Task: Map existing folders in Results/ (e.g., optim_collocation, comparison_pinn_grid) to the new numeric goal categories.
- [x] Task: Move artifacts into experiments/ and add the documented script copy for each migrated case.
- [x] Task: Add specific notes to optim_collocation regarding its high_res/std_res sub-structure.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Migration' (Protocol in workflow.md)

## Phase 4: Final Verification and Cleanup
Ensure the system is fully functional and the root directory is clean.

- [ ] Task: Run a sample experiment from `Heat2D_main.py` and verify artifact generation in the correct `experiments/` subfolder.
- [ ] Task: Verify that all script copies in `experiments/` contain the correct headers.
- [ ] Task: Remove the legacy `Heat2D/Results/` directory.
- [ ] Task: Conductor - User Manual Verification 'Phase 4: Cleanup' (Protocol in workflow.md)
