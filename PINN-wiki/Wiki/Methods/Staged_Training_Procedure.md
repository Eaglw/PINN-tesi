# Method: Staged Training Procedure

## Overview
The Staged Training Procedure (also known as Decoupled Training) is a robust optimization strategy for complex physical systems where multiple neural networks are coupled. In the context of Viscoelastic PINNs, it prevents the instability caused by high-magnitude residuals from one field (e.g., stress) disrupting the learning of another (e.g., velocity).

## Implementation Details
The procedure splits the total training epochs into distinct stages, selectively freezing and unfreezing parts of the `ViscoelasticCombinedModel`.

### Stage 1: Kinematics (Flow Field)
- **Active Networks**: `model_psi`, `model_p`
- **Frozen Networks**: `model_tau`
- **Objective**: Learn the velocity field and pressure distribution. By freezing the stress network, the optimizer focuses on solving the Navier-Stokes part of the problem without being distracted by non-linear constitutive residuals.
- **Duration**: ~40% of total epochs.

### Stage 2: Constitutive (Stress Field)
- **Active Networks**: `model_tau`
- **Frozen Networks**: `model_psi`, `model_p`
- **Objective**: Given the fixed velocity field from Stage 1, learn the corresponding polymeric stress components ($\tau_{xx}, \tau_{xy}, \tau_{yy}$) that satisfy the Oldroyd-B constitutive equations.
- **Duration**: ~40% of total epochs.

### Stage 3: Full Coupled (Refinement)
- **Active Networks**: `model_psi`, `model_p`, `model_tau`
- **Objective**: Jointly optimize all fields. This allows the networks to "talk" to each other and refine the global solution, ensuring all physical laws are simultaneously satisfied.
- **Duration**: ~20% of total epochs.

## Control Logic
The strategy is controlled via the `STAGED_TRAINING` flag in `Viscoelastic_main.py` and the helper function `set_model_trainable()` in `Viscoelastic_PINN.py`.

## Advantages for Thesis
1. **Convergence Stability**: Avoids gradient competition in the early stages.
2. **Modular Verification**: Allows checking if the velocity field is correct before committing to complex stress calculations.
3. **Hyperparameter Isolation**: Makes it easier to tune learning rates for specific components.

## Related
- [[ViscoelasticNet]]
- [[Staged_Precision_Strategy]]
- [[Oldroyd_B_Model]]
