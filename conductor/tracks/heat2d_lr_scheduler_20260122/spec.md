# Specification: Step Decay LR Scheduler for Heat2D

## Overview
Implement a Step Decay learning rate scheduler for all Heat2D models and integrate it as a configurable option in the grid search within `Heat2D_main.py`. This includes updating training scripts to support schedulers and enhancing `results.csv` logging to reflect the LR strategy.

## Functional Requirements
- **Grid Search Integration:**
    - Update `Heat2D_main.py` to include a `lr_strategy` parameter in the experiment grid.
    - Options: `fixed` (standard constant LR) or `step_decay`.
- **Step Decay Implementation:**
    - Integrate `torch.optim.lr_scheduler.StepLR` into the training loops of `Heat2D_NN.py`, `Heat2D_NN_griglia.py`, and `Heat2D_PINN.py`.
    - **Step size:** `0.25 * total_epochs`.
    - **Decay factor (gamma):** `0.5`.
    - The scheduler should only be active if `lr_strategy == 'step_decay'`.
- **Results Logging:**
    - Update logging logic to capture the learning rate behavior.
    - Modify `results.csv`:
        - For `fixed`: log the single value (e.g., `0.001`).
        - For `step_decay`: log the range as `[initial_lr -> final_lr]` (e.g., `[0.001 -> 0.000125]`).

## Non-Functional Requirements
- **Modular Design:** Ensure training scripts receive the strategy as an argument to remain decoupled from the main loop logic.
- **Precision:** Maintain `torch.float64` precision.

## Acceptance Criteria
- [ ] `Heat2D_main.py` successfully iterates through both `fixed` and `step_decay` strategies.
- [ ] Training scripts correctly apply the `StepLR` logic when requested.
- [ ] `results.csv` accurately records either the fixed value or the range based on the strategy used.

## Out of Scope
- Other scheduler types (Cosine, etc.).
- Changes to CSTR modules.
