# Specification: Point Overlap Verification in Weighted Heat2D

## Overview
This track aims to integrate safety checks in `Heat2D_weighted_main.py` to ensure that training and collocation point sets do not contain overlapping points, maintaining consistency with the standard `Heat2D_main.py` implementation.

## Functional Requirements
- **Overlap Detection:** Implement checks to verify that points within and between key datasets are sufficiently separated.
- **Target Sets:**
    - `xy_master_grid` (Collocation points)
    - `xy_pinn_data` (Training data points)
    - `xy_master_boundary` (Boundary points)
    - Combined set: `xy_master_grid` + `xy_pinn_data` + `xy_master_boundary`
- **Validation Logic:** Use the existing `check_overlaps` utility from `func.sampling_utils`.
- **Error Handling:** If an overlap is detected (minimum distance < 1e-7), the script must print a warning and terminate execution (using `sys.exit(1)` or similar) to prevent training on corrupted or suboptimal data distributions.

## Non-Functional Requirements
- **Consistency:** The implementation should mirror the structure and logic used in `Heat2D_main.py`.
- **Performance:** Verification should occur once before the grid search begins to avoid redundant checks.

## Acceptance Criteria
- [ ] `Heat2D_weighted_main.py` imports `check_overlaps` from `func.sampling_utils`.
- [ ] The script performs overlap checks immediately after the master sets are defined.
- [ ] The script successfully proceeds if no overlaps are found.
- [ ] The script terminates with a clear message if an overlap is detected in any of the target sets.

## Out of Scope
- Modifying the sampling algorithms themselves (e.g., `filter_and_refill`).
- Adding overlap checks to other scripts not mentioned.
