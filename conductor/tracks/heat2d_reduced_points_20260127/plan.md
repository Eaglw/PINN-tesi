# Implementation Plan: Reduced Points Grid Search (Heat2D Direct)

## Phase 1: Infrastructure & Data Migration
Update the logging system and prepare the results file for the new schema.

- [x] Task: Update `Heat2D/results.csv` schema. Add `n_points` column and migrate existing data (set to 2000).
- [x] Task: Modify `func/logging_utils.py` to support `n_points` in `update_results_csv`.
- [x] Task: Create `Heat2D/experiments_reduced_points/` directory.
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Infrastructure & Data Migration' (Protocol in workflow.md)

## Phase 2: Implementation of Reduced Points Runner
Create the script to handle the reduced dataset and the specific grid search.

- [ ] Task: Create `Heat2D/Heat2D_reduced_main.py` based on `Heat2D/Heat2D_main.py`.
- [ ] Task: Implement reduced dataset generation in `Heat2D/Heat2D_reduced_main.py` (300 internal, 200 boundary).
- [ ] Task: Configure Grid Search in `Heat2D/Heat2D_reduced_main.py` (Exclude 80x6, include all other hparams).
- [ ] Task: Ensure logging in `Heat2D/Heat2D_reduced_main.py` correctly reports `n_points` as ~500.
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Implementation of Reduced Points Runner' (Protocol in workflow.md)

## Phase 3: Execution & Verification
Run the experiments and verify the results.

- [ ] Task: Execute the reduced points grid search using `Heat2D/Heat2D_reduced_main.py`.
- [ ] Task: Verify that results are correctly saved in `Heat2D/experiments_reduced_points/`.
- [ ] Task: Verify that `Heat2D/results.csv` contains the new entries with `n_points` ≈ 500.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Execution & Verification' (Protocol in workflow.md)
