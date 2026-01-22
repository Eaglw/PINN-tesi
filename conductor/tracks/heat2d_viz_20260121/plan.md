# Implementation Plan - Heat2D Visualization Improvements

This plan outlines the steps to refine the visualization output for Heat2D experiments, including a 2-column final result plot and a 2x2 unified comparison plot.

## Phase 1: Analysis and Infrastructure (Preparation) [checkpoint: cc91eeb]
- [x] Task: Analyze current plotting implementation in `Heat2D/src/` and `func/graphic_func.py` to identify the best insertion points for changes.
- [x] Task: Create a utility function to extract hyperparameters (architecture, epochs, activation) from experiment paths or metadata.
- [x] Task: Conductor - User Manual Verification 'Phase 1: Analysis and Infrastructure (Preparation)' (Protocol in workflow.md)

## Phase 2: Refactor Final Result Visualization [checkpoint: e4db933]
- [x] Task: Modify `Heat2D_PINN.py` (and related scripts) to pass training points (physics and data) to the plotting function.
- [x] Task: Update the `final_result.png` generation logic to produce a 2-column layout (Solution with Points vs Relative Error).
- [x] Task: Ensure training points are plotted with distinct colors and a clear legend.
- [x] Task: Verify the new `final_result.png` layout across different Heat2D runs.
- [x] Task: Conductor - User Manual Verification 'Phase 2: Refactor Final Result Visualization' (Protocol in workflow.md)

## Phase 3: Unified Comparison Plotting
- [ ] Task: Implement a new script or function in `Heat2D/src/` (e.g., `comparison_utils.py`) to generate the 2x2 error map comparison.
- [ ] Task: Implement individual colorbars for each subplot in the 2x2 grid.
- [ ] Task: Add the hyperparameter metadata title to the comparison plot.
- [ ] Task: Update the main execution flow to call the unified comparison generator after experiments are complete.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Unified Comparison Plotting' (Protocol in workflow.md)

## Phase 4: Integration and Cleanup
- [ ] Task: Ensure the updated visualization logic is applied consistently to all Heat2D scripts (`Heat2D_NN.py`, `Heat2D_PINN.py`, etc.).
- [ ] Task: Verify that existing `results.csv` logging is compatible with the new metadata requirements for titles.
- [ ] Task: Perform a final test run of a full experiment suite to confirm all plots are generated correctly.
- [ ] Task: Conductor - User Manual Verification 'Phase 4: Integration and Cleanup' (Protocol in workflow.md)
