# Implementation Plan - Standardize Heat2D Point Distribution (Internal)

## Phase 1: Preparation
- [x] Task: Analyze `Heat2D/Heat2D_main.py` to identify all current point generation locations.
- [x] Task: Conductor - User Manual Verification 'Phase 1' (Protocol in workflow.md)

## Phase 2: Implementation in Main Script
- [x] Task: Refactor the "GENERAZIONE GRIGLIE" section in `Heat2D/Heat2D_main.py`.
    - [x] Sub-task: Set a fixed master seed for reproducibility.
    - [x] Sub-task: Generate **Master Grid Points** (1600 points, strictly internal).
    - [x] Sub-task: Generate **Master Random Points** (1600 points, strictly internal).
    - [x] Sub-task: Generate **Master Boundary Points** (400 points, 100 per side).
- [x] Task: Refactor the individual experiment configurations in the main loop.
    - [x] Sub-task: Update **Case 0 (NN Random)** to use 1600 random + 400 boundary.
    - [x] Sub-task: Update **Case 1 (NN Grid)** to use 1600 grid + 400 boundary.
    - [x] Sub-task: Update **Case 2 (PINN Data+Phys)** to use 1600 grid for physics, 400 boundary for BC, and 1000 random subset for data.
    - [x] Sub-task: Update **Case 3 (PINN Pure Phys)** to use 1600 grid for physics and 400 boundary for BC.
- [x] Task: Conductor - User Manual Verification 'Phase 2' (Protocol in workflow.md)

## Phase 3: Verification
- [x] Task: Run `Heat2D/Heat2D_main.py` for 1 epoch for all cases and verify printed point counts/shapes.
- [x] Task: Conductor - User Manual Verification 'Phase 3' (Protocol in workflow.md)
