# Implementation Plan - Heat2D Results and Experiments Cleanup

This plan covers the creation of a maintenance script to clean up `Heat2D` results and experiment folders based on an epoch threshold (< 10,000).

## Phase 1: Script Development & Logic Verification [checkpoint: 541bbaf]

- [x] Task: Create `Heat2D/cleanup_experiments.py` with skeleton logic
- [x] Task: Implement CSV parsing and filtering logic (Epochs < 10,000)
- [x] Task: Implement directory mapping logic (searching `Heat2D/experiments/`)
- [x] Task: Implement Dry Run and confirmation prompt
- [x] Task: Implement file writing (updating CSV) and directory removal
- [x] Task: Conductor - User Manual Verification 'Phase 1: Script Development' (Protocol in workflow.md)

## Phase 2: Execution and Cleanup

- [x] Task: Execute the script in Dry Run mode and verify targeted items
- [x] Task: Run the cleanup script and confirm deletion
- [x] Task: Verify that `results.csv` no longer contains filtered entries
- [x] Task: Verify that identified directories have been removed from the filesystem
- [x] Task: Update script to scan filesystem for orphan experiment folders (< 10k epochs)
- [x] Task: Execute updated script to clean orphan folders
- [x] Task: Conductor - User Manual Verification 'Phase 2: Execution and Cleanup' (Protocol in workflow.md)
