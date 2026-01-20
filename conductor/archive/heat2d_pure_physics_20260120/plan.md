# Implementation Plan: Pure Physics-Informed 2D Heat Transfer

Implement a pure physics-informed training mode (no experimental data) within the `Heat2D` module, controlled by a numeric `GOAL` variable in `Heat2D_main.py`. To keep the main script clean, specific configurations for this mode will be isolated in a separate helper script.

## Phase 1: Analysis and Refactoring Preparation
Understand the existing numeric `GOAL` structure in `Heat2D_main.py` and prepare the offloaded configuration logic.

- [x] Task: Analyze `Heat2D/Heat2D_main.py` to identify existing `GOAL` values and the next available integer.
- [x] Task: Identify the exact parameters and objects (loss weights, data loaders, paths) that need to be configured differently for the new goal.
- [x] Task: Conductor - User Manual Verification 'Phase 1: Analysis' (Protocol in workflow.md)

## Phase 2: Implementation of Configuration Script
Create the separate configuration script and integrate it with minimal impact on the main execution flow.

- [x] Task: Create `Heat2D/pure_physics_setup.py` (or similar).
    - Implement a function (e.g., `get_pure_physics_config()`) that returns the specific data loaders (None/Empty), loss weights, and directory paths for this case.
- [x] Task: Modify `Heat2D/Heat2D_main.py`:
    - Define the new `GOAL` constant.
    - Import the setup script.
    - Add a concise conditional block: `if GOAL == NEW_CASE: config = get_pure_physics_config()`.
- [x] Task: Ensure the training loop respects the returned configuration (e.g., handles `None` for data loaders gracefully).
- [x] Task: Conductor - User Manual Verification 'Phase 2: Implementation' (Protocol in workflow.md)

## Phase 3: Evaluation and Visualization
Ensure the results of the pure physics run are correctly logged and visualized for comparison.

- [x] Task: Create a new directory `Results/pure_physics/` (managed by the setup script).
- [x] Task: Verify that `history_tracker.py` correctly logs the reduced loss set (PDE + BC) based on the config.
- [x] Task: Ensure the plotting functions called by main can handle the pure physics context (e.g., plotting without data scatter points).
- [x] Task: Conductor - User Manual Verification 'Phase 3: Evaluation' (Protocol in workflow.md)

## Phase 4: Final Verification
Verify the end-to-end execution and backward compatibility.

- [x] Task: Run `Heat2D/Heat2D_main.py` with the new `GOAL` and verify training convergence and output generation.
- [x] Task: Briefly test an existing `GOAL` mode to ensure no regressions were introduced.
- [x] Task: Conductor - User Manual Verification 'Phase 4: Final Verification' (Protocol in workflow.md)
