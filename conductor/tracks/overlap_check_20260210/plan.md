# Plan: Point Overlap Verification in Weighted Heat2D

## Phase 1: Preparation [checkpoint: 48b6b8a]
- [x] Task: Update imports in `Heat2D_weighted_main.py`
    - [x] Import `check_overlaps` from `func.sampling_utils`
    - [x] Ensure `sys` is imported for script termination
- [x] Task: Conductor - User Manual Verification 'Phase 1: Preparation' (Protocol in workflow.md)

## Phase 2: Implementation
- [ ] Task: Implement Overlap Verification Logic
    - [ ] Locate the section after master sets generation
    - [ ] Add `check_overlaps` calls for `xy_master_grid`, `xy_pinn_data`, and `xy_master_boundary`
    - [ ] Add a combined check for the full PINN set
    - [ ] Implement conditional `sys.exit(1)` if any check fails
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Implementation' (Protocol in workflow.md)

## Phase 3: Verification
- [ ] Task: Verify successful execution
    - [ ] Run `Heat2D_weighted_main.py` and confirm it passes the checks and starts training
- [ ] Task: Verify failure handling (Negative Test)
    - [ ] Temporarily modify the script to inject overlapping points
    - [ ] Confirm the script prints the warning and terminates correctly
    - [ ] Revert the temporary modification
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Verification' (Protocol in workflow.md)
