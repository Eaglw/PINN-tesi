# Implementation Plan: Heat2D Experiment Logging System

This plan outlines the steps to implement a centralized logging system for Heat2D experiments, capturing metadata and performance metrics in a CSV format.

## Phase 1: Metric Calculation and Logging Infrastructure

- [ ] Task: Create `func/logging_utils.py` and implement `compute_metrics` (L2 & Max Relative Error) and `update_results_csv` (CSV handling).
- [ ] Task: Create a temporary test script `func/test_logging.py` to verify CSV formatting, file creation, and metric calculation logic.
- [ ] Task: Run the test script to confirm functionality.
- [ ] Task: Delete `func/test_logging.py` after successful verification.
- [ ] Task: Conductor - User Manual Verification 'Phase 1' (Protocol in workflow.md)

## Phase 2: Integration and Validation

- [ ] Task: Import `update_results_csv` in `Heat2D/Heat2D_main.py`.
- [ ] Task: Integrate `update_results_csv` calls into each experimental case (NN Random, NN Grid, PINNs) within `Heat2D/Heat2D_main.py`, passing all required metadata and history objects.
- [ ] Task: Perform a "smoke test" run of `Heat2D/Heat2D_main.py` with significantly reduced epochs (e.g., 100) and a single case to verify `results.csv` population.
- [ ] Task: Verify the generated `Heat2D/results.csv` content matches the specification (columns, order, precision).
- [ ] Task: Conductor - User Manual Verification 'Phase 2' (Protocol in workflow.md)

## Phase 3: Final Cleanup

- [ ] Task: Final code review to ensure adherence to style guides and non-functional requirements.
- [ ] Task: Conductor - User Manual Verification 'Phase 3' (Protocol in workflow.md)
