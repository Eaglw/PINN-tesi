# Implementation Plan - Grid-based NN Training vs Random Sampling

## Phase 1: Scaffolding and Integration [checkpoint: c994da9]
- [x] Task: Create `Heat2D/Heat2D_NN_griglia.py`
    - [x] Port logic from `Heat2D_NN.py` but replace random sampling with `torch.meshgrid` grid generation (matching `Heat2D_PINN.py`).
    - [x] Implement a `train_modelNN_griglia` function similar to `train_modelNN`.
- [x] Task: Update `Heat2D/Heat2D_main.py` for integration
    - [x] Add a new goal (e.g., `goal=5`) for NN Griglia.
    - [x] Add the import and logic to call `train_modelNN_griglia` when selected.
- [x] Task: Conductor - User Manual Verification 'Scaffolding and Integration' (Protocol in workflow.md)

## Phase 2: Implementation and Verification
- [ ] Task: Implement Grid Sampling logic
    - [ ] Ensure the number of grid points in `Heat2D_NN_griglia.py` matches the density/count of random points in `Heat2D_NN.py` for fair comparison.
    - [ ] Verify that boundary points are handled consistently with the grid approach.
- [ ] Task: Implement Comparison Logic in `Heat2D_main.py`
    - [ ] Store history/results from both `goal=0` and `goal=5` runs.
    - [ ] Create a comparison function to plot overlapping loss curves.
    - [ ] Create a comparison function for point-wise error maps.
- [ ] Task: Conductor - User Manual Verification 'Implementation and Verification' (Protocol in workflow.md)

## Phase 3: Final Integration and Testing
- [ ] Task: Verify Goal Execution
    - [ ] Run `Heat2D_main.py` with `goal=[5]` and verify output in `Results/`.
- [ ] Task: Verify Comparative Run
    - [ ] Run `Heat2D_main.py` with `goal=[0, 5]` and verify that the comparison plots and error maps are correctly generated.
- [ ] Task: Quality Gate Check
    - [ ] Ensure all code follows project style guidelines.
    - [ ] Ensure no unnecessary libraries were added.
- [ ] Task: Conductor - User Manual Verification 'Final Integration and Testing' (Protocol in workflow.md)
