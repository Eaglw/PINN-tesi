# Specification: Dynamic and Static Weighting Strategy in Heat2D

## Overview
This track aims to implement both a coherent **static weighting** strategy and a **dynamic weighting** strategy (specifically Learning Rate Annealing) for the physics and data loss components in `Heat2D_weighted_main.py`. The goal is to correct loss disparities and evaluate if dynamic weighting offers superior performance over static or unweighted approaches. The results will be logged with a clear distinction in the `Loss_Weight` column of `results.csv`.

## Functional Requirements
- **Static Weighting:**
    - Implement a run using fixed weights: `BC=1.0, PHYS=10.0, DATA=100.0`.
    - Apply this to both Goal 2 (Data+Physics) and Goal 3 (Pure Physics) configurations.
- **Dynamic Weighting (Learning Rate Annealing):**
    - Implement the "Learning Rate Annealing" algorithm (Wang et al.) to dynamically adjust `PHYS` and `DATA` weights during training.
    - The boundary weight (`BC`) should generally remain fixed (or serve as the anchor).
    - Update weights at a defined frequency (e.g., every epoch or every N iterations).
- **Grid Search Integration:**
    - The existing grid search loop in `Heat2D_weighted_main.py` must be expanded to include these two weighting strategies as distinct experimental "modes" or "configurations".
- **Logging:**
    - Update `results.csv` logging to include a `Loss_Weight` column.
    - Value for static run: `"BC=1-PHYS=10-DATA=100"`.
    - Value for dynamic run: `"Dynamic-Annealing"`.

## Non-Functional Requirements
- **Code Structure:** The implementation should be modular. The dynamic weighting logic should ideally be encapsulated in a class or function within `src/` to keep `Heat2D_weighted_main.py` clean.
- **Performance:** Dynamic weight updates require gradient computations; ensure this doesn't drastically degrade training speed (compute gradients only when necessary for weight updates).

## Acceptance Criteria
- [ ] `Heat2D_weighted_main.py` runs both Static and Dynamic weighting experiments for each grid search configuration.
- [ ] Static weights are applied correctly as `BC=1.0, PHYS=10.0, DATA=100.0`.
- [ ] Dynamic weighting (LR Annealing) is implemented and active during the "Dynamic" run.
- [ ] `results.csv` accurately reflects the weighting strategy in the `Loss_Weight` column.
- [ ] The script completes without error for both Goal 2 and Goal 3.

## Out of Scope
- Implementing other dynamic weighting strategies (SoftAdapt, etc.).
- modifying the unweighted baseline logic (unless necessary for refactoring).
