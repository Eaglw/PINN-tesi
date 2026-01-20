# Specification: Pure Physics-Informed 2D Heat Transfer

## Overview
Implement a "Pure PINN" training mode for the 2D Heat Transfer problem. This mode will use a loss function strictly consisting of physics residuals (PDE) and boundary conditions (BCs), with zero reliance on experimental or labeled data. The execution will be controlled via the existing `Heat2D/Heat2D_main.py` script by adding a new numeric case to the `GOAL` configuration variable.

## Functional Requirements
- **Refactor `Heat2D_main.py`:**
    - Identify the existing numeric `GOAL` constants/logic.
    - Define a new numeric constant (e.g., `GOAL = 4` or the next available integer) representing "PURE_PHYSICS".
    - Implement conditional logic in the training setup: when this goal is selected, bypass data loading/loss calculation and optimize $L_{total} = \lambda_p L_{physics} + \lambda_{BC} L_{BC}$.
- **Physics-Only Logic:** Ensure the training loop correctly adapts to the selected goal, disabling data-related computations.
- **Integration:** Reuse the existing `Heat2D_NN` architecture and physics definitions.
- **Evaluation & Comparison:**
    - Log training metrics specific to the active mode.
    - Save the final model state to `models/` with a distinct naming convention (e.g., `_pure_physics`).
    - Generate visualization plots in `Results/` (specifically under a new subfolder like `pure_physics`) comparing the solution against the reference.

## Non-Functional Requirements
- **Precision:** Use `torch.float64`.
- **Performance:** Support GPU acceleration (`cuda`).
- **Standardization:** Use `func/history_tracker.py` and `func/graphic_func.py`.
- **Consistency:** The configuration style must match the existing numeric `GOAL` pattern in the file.

## Acceptance Criteria
1. `Heat2D/Heat2D_main.py` runs successfully when `GOAL` is set to the new numeric value.
2. The training loop runs without using any interior data points for loss.
3. Existing modes (other `GOAL` numbers) continue to function unchanged.
4. Comparative plots are generated and saved correctly for the pure physics run.

## Out of Scope
- Implementation of inverse problems for this specific task.
