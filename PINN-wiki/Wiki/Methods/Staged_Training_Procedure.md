# Method: Staged Training Procedure

## Overview
The Staged Training Procedure (also known as Decoupled Training) is a robust optimization strategy for complex physical systems where multiple neural networks are coupled. In Viscoelastic PINNs, it prevents instability by decoupling the learning of kinematics (velocity/pressure) from the non-linear constitutive stress field.

## Implementation (Standard Architecture)
As of May 2026, the procedure is integrated directly into the training core (`train_ViscoelasticPINN`), simplifying the user interface and ensuring consistency.

### Phase 1: Kinematics & Rheology (Adam - First 50%)
- **Active Networks**: `model_psi`, `model_tau`
- **Frozen Networks**: `model_p`
- **Active PDE Losses**: `constitutive` (Oldroyd-B ON), `momentum: 0.0` (Navier-Stokes OFF).
- **Active Boundary Conditions**: `['u', 'v', 'txx', 'txy', 'tyy']` (Pressure BCs excluded).
- **Objective**: Establish a stable velocity profile ($\psi$) from boundary/internal data while simultaneously discovering the corresponding polymeric stress field ($\boldsymbol{\tau}$) via the Oldroyd-B constitutive laws, completely isolated from pressure fluctuations.

### Phase 2: Dynamics (Adam - Second 50%)
- **Active Networks**: `model_psi`, `model_p` (Note: `model_psi` must be active to overcome the Helmholtz-Hodge limit, see [[Pressure_Stress_Decoupling#The Helmholtz-Hodge Pressure Inference Limit]]).
- **Frozen Networks**: `model_tau`
- **Active PDE Losses**: `momentum` ON, `constitutive` ON (All PDEs active).
- **Active Boundary Conditions**: `['u', 'v', 'p']` (Stress BCs excluded).
- **Objective**: Learn the pressure distribution $p(x,y)$ required to balance the Navier-Stokes momentum equations.
- **Key Challenges & Optimizations**:
  * **Vanishing Gradient Cascade**: Because the last layer of `model_p` is initialized to zero (`initialize_last_layer_zero()`), the spatial gradient of pressure $\nabla p$ is initially zero. Since all momentum loss gradients flow through $\nabla p$ for `model_p`, this freezes the gradients of the internal layers of `model_p` at the very first step if velocity is completely frozen. Allowing `model_psi` to train and backpropagate provides the necessary updates to break this cascade.
  * **Divergence of Stress Precomputation**: Since `model_tau` is frozen, its divergence $\nabla \cdot \boldsymbol{\tau}$ is constant. It is precomputed once at the beginning of Phase 2 (and re-evaluated when switching to FP64 in Phase 3) and cached. This avoids redundant forward passes through `model_tau` and three spatial derivative computations at each epoch, yielding a **25% - 30% speedup** in Phase 2.

### Phase 3: Full Coupled Refinement (L-BFGS)
- **Active Networks**: **All Networks** (`psi`, `p`, `tau` fully unfrozen).
- **Precision Mode**: Switch to **FP64** (Double Precision).
- **Active PDE & BCs**: All PDE residuals and all 6 boundary conditions (`u, v, p, txx, txy, tyy`) are active simultaneously.
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
- [[Viscoelastic_Training]]
- [[Staged_Precision_Strategy]]
- [[Oldroyd_B_Model]]
