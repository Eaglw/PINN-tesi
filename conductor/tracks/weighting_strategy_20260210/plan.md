# Plan: Dynamic and Static Weighting Strategy in Heat2D

## Phase 1: Implementation of Dynamic Weighting (LR Annealing) [checkpoint: 434e768]
- [x] Task: Update `Heat2D_PINN.py` to support dynamic weighting (LR Annealing).
    - [x] Modify `train_modelPINN` to include an `update_weights_every` parameter.
    - [ ] Implement the gradient-based weight update logic (Wang et al.):
        - Calculate grad norms for BC loss ($\bar{
abla}_{	heta} \mathcal{L}_{bc}$).
        - Calculate grad norms for Physics loss ($\hat{
abla}_{	heta} \lambda_{p} \mathcal{L}_{p}$).
        - Update $\lambda_{p}$ using the ratio of moving averages of these norms.
        - Repeat for Data loss if applicable.
- [x] Task: Conductor - User Manual Verification 'Phase 1: Implementation of Dynamic Weighting (LR Annealing)' (Protocol in workflow.md)

## Phase 2: Refactoring Heat2D_weighted_main.py for Comparative Runs
- [ ] Task: Expand the Grid Search loop to handle different weighting strategies.
    - [ ] Add `weighting_options = ['static', 'dynamic']` to the grid search.
    - [ ] Configure `STATIC_WEIGHTS = {'bc': 1.0, 'physics': 10.0, 'data': 100.0}`.
    - [ ] Implement logic to switch between `loss_weights` dictionaries and `dynamic_weighting` flags in the training call.
- [ ] Task: Update the logging mechanism.
    - [ ] Ensure `Loss_Weight` column correctly records `"BC=1-PHYS=10-DATA=100"` or `"Dynamic-Annealing"`.
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Refactoring Heat2D_weighted_main.py for Comparative Runs' (Protocol in workflow.md)

## Phase 3: Verification, Documentation, and Integration
- [ ] Task: Verify the "Static" weighting run.
    - [ ] Execute a limited run (few epochs) of `Heat2D_weighted_main.py` in static mode.
    - [ ] Confirm `results.csv` logging and loss balancing.
- [ ] Task: Verify the "Dynamic" weighting run.
    - [ ] Execute a limited run in dynamic mode.
    - [ ] Monitor weight evolution (logs or print statements) to ensure LR Annealing is active.
- [ ] Task: Create Documentation Note.
    - [ ] Create `notes/Dynamic_Weighting_Implementation.md`.
    - [ ] Document the mathematical foundation of LR Annealing and the specifics of its implementation in this project.
- [ ] Task: Final full-scale test.
    - [ ] Run a representative configuration to ensure stability and accuracy.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Verification, Documentation, and Integration' (Protocol in workflow.md)
