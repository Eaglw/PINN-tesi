# Method: ViscoelasticNet

## Overview
ViscoelasticNet is a deep learning framework designed to solve both forward and inverse problems in viscoelastic fluid mechanics. It extends the standard PINN approach by incorporating constitutive laws directly into the loss function.

## Technical Implementation
Key features of the current repository implementation:
- **Stream Function Formulation**: Velocity is derived from a stream function network ($u = \psi_y, v = -\psi_x$), ensuring divergence-free flow ($\nabla \cdot \mathbf{u} = 0$) by construction.
- **Multi-Network Architecture**: Uses `ViscoelasticCombinedModel` to unify separate networks for:
    - **Stream Function ($\psi$)**: Scalar output.
    - **Pressure ($p$)**: Scalar output.
    - **Stress ($\tau$)**: 3-output network for $\tau_{xx}, \tau_{xy}, \tau_{yy}$.
- **Physical Loss**:
    - **Momentum**: Couples velocity, pressure, and the divergence of the extra stress tensor.
    - **Oldroyd-B**: Implements the upper-convected constitutive equation.
- **Staged Training**: Transitions from Adam (exploration) to L-BFGS (refinement) while switching from `float32` to `float64` for numerical stability.

## References
- [[Thakur_et_al_ViscoelasticNet]]
- [[Oldroyd_B_Model]]
- [[Viscoelasticity]]
- [[Dynamic_Weighting]]
- [[Staged_Precision_Strategy]]
