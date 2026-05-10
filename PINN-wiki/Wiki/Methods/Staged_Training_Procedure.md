# Method: Staged Training Procedure

## Overview
The Staged Training Procedure (also known as Decoupled Training) is a robust optimization strategy for complex physical systems where multiple neural networks are coupled. In Viscoelastic PINNs, it prevents instability by decoupling the learning of kinematics (velocity/pressure) from the non-linear constitutive stress field.

## Implementation (Standard Architecture)
As of May 2026, the procedure is integrated directly into the training core (`train_ViscoelasticPINN`), simplifying the user interface and ensuring consistency.

### Phase 1: Kinematics (Adam - First 50%)
- **Active**: `model_psi`, `model_p`
- **Frozen**: `model_tau`
- **Objective**: Establish a stable flow field (velocity and pressure) without the high-gradient interference of polymeric stress residuals.

### Phase 2: Constitutive (Adam - Second 50%)
- **Active**: `model_tau`
- **Frozen**: `model_psi`, `model_p`
- **Objective**: Compute the polymeric stress tensor components ($\tau_{xx}, \tau_{xy}, \tau_{yy}$) corresponding to the fixed flow field from Phase 1, solving the Oldroyd-B constitutive equations.

### Phase 3: Full Coupled (L-BFGS Refinement)
- **Active**: **All Networks** (`psi`, `p`, `tau`)
- **Precision**: Switch to **FP64** (Double Precision).
- **Objective**: Jointly optimize all fields. This global refinement ensures that the final solution satisfies both the momentum/continuity equations and the constitutive laws simultaneously with high numerical precision.

## Control Logic
The strategy is enabled via the `staged_training=True` flag in the training call. The function internally manages:
1. `set_model_trainable()` calls to freeze/unfreeze parameters.
2. Optimizer and scheduler resets at the phase transition point.
3. Total unfreezing before the L-BFGS precision switch.

## Advantages
1. **Convergence Stability**: Decouples competitive gradients during the early exploration phase.
2. **Implementation Robustness**: Reduces boilerplate code in experiment scripts and prevents common errors in manual stage management.
3. **Physical Grade Accuracy**: L-BFGS joint optimization ensures the coupled system reaches a physically consistent state.

## Related
- [[ViscoelasticNet]]
- [[Staged_Precision_Strategy]]
- [[Oldroyd_B_Model]]
