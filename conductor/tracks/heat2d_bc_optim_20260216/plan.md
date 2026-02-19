# Implementation Plan: Heat2D Boundary Conditions Optimization

## Phase 1: Implementation of Optimized Sampling [checkpoint: ]
- [x] Task: Update boundary point generation logic in `Heat2D/Heat2D_main.py` and `Heat2D/Heat2D_weighted_main.py` to use 50 points per side (200 total).
- [x] Task: Implement a 0.02 safety margin to exclude points near the four corners (0,0), (1,0), (0,1), (1,1).
- [x] Task: Replace `soluzione_analitica` for boundary targets with exact physical constants (0.0 for Left, Bottom, Top; 1.0 for Right).
- [x] Task: Verify that `check_overlaps` confirms the new distribution and the absence of corner points.
- [~] Task: Conductor - User Manual Verification 'Implementation of Optimized Sampling' (Protocol in workflow.md)
