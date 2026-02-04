# Implementation Plan: Heat2D Point Overlap Prevention

Ensuring spatial disjointness between collocation and data points, and maintaining a safety margin from boundaries to enhance training stability.

## Phase 1: Sampling Utilities and Unit Testing
Focuses on creating robust, reusable logic for constrained point sampling.

- [x] Task: Create `Heat2D/src/sampling_utils.py` containing:
    - `generate_internal_points(num_points, Lx, Ly, margin=1e-5)`: Random sampling within margins.
    - `generate_grid_points(Nx, Ny, Lx, Ly, margin=1e-5)`: Grid sampling shifted away from boundaries.
    - `filter_and_refill(primary_set, secondary_set_generator, target_count, d_min=1e-4)`: Logic to ensure disjointness and maintain target count.
- [x] Task: Create `Heat2D/tests/test_sampling_utils.py` and implement:
    - Test boundary safety margin enforcement.
    - Test Euclidean distance filtering between sets.
    - Test iterative regeneration to reach target counts.
- [~] Task: Conductor - User Manual Verification 'Sampling Utilities and Unit Testing' (Protocol in workflow.md)

## Phase 2: Integration and Refactoring
Integrating the new utilities into the main execution flow.

- [ ] Task: Refactor `Heat2D/Heat2D_main.py` point generation logic:
    - Update Master Grid Set generation using `sampling_utils`.
    - Update Master Random Set generation using `sampling_utils`.
    - Implement `filter_and_refill` for the `PINN Data+Phys` case (Data points must not overlap Collocation points).
- [ ] Task: Ensure consistency in `Heat2D_NN_griglia.py` and other source files if they perform independent sampling.
- [ ] Task: Conductor - User Manual Verification 'Integration and Refactoring' (Protocol in workflow.md)

## Phase 3: Diagnostics and Training Verification
Final validation through visualization and empirical testing.

- [ ] Task: Implement a diagnostic plot in `Heat2D_main.py` (or a separate script) that highlights overlapping points (if any) and boundary margins.
- [ ] Task: Run a benchmark comparison of `PINN_DataPhys` before and after the change to verify that training remains stable or improves.
- [ ] Task: Conductor - User Manual Verification 'Diagnostics and Training Verification' (Protocol in workflow.md)
