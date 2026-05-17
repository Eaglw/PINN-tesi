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
- **Staged Training (Decoupling)**: Implements a 3-stage training strategy (Kinematics $\to$ Constitutive $\to$ Full Coupled) to stabilize convergence. See [[Staged_Training_Procedure]] for details.
- **Precision Switching**: Transitions from Adam (exploration) to L-BFGS (refinement) while switching from `float32` to `float64`. See [[Staged_Precision_Strategy]].

## Recent Updates: Semi-Inverse Strategy (Goal 1)
To fully align with the original ViscoelasticNet methodology, a `semi_inverse` mode (Goal 1) has been implemented:
- **Supervision on Velocity**: The internal data loss is computed strictly as $MSE(u_{pred}, u_{obs}) + MSE(v_{pred}, v_{obs})$, driving the stream function $\psi$ purely via its derivatives.
- **Variance Scaling**: All loss components (PDE, BC, Data) are normalized by the variance of the reference velocity field ($max(\sigma^2_u, 1e-8)$) to balance gradients.
- **Optimization Enhancements**: Utilizes mini-batching ($N_{int}=256, N_{bc}=64$) to stochasticize the gradient descent, coupled with a Cosine Annealing learning rate scheduler and an Adam `eps=1e-7` for robust exploration.

### Future Extension: `inverse_dense` (Full-Field Inverse Problem)
To extend the framework to a fully inverse scenario ("inverse_dense") matching ViscoelasticNet with dense PIV/CFD data:
1. **Full-Field Collocation**: Replace the sparse, randomly sampled collocation points with the dense, uniform PIV/CFD grid ($xy\_grid\_flat$).
2. **Global Loss Calculation**: Compute the data loss across the entire spatial domain rather than a subset.
3. **Unknown Parameter Identification**: Define physical parameters (e.g., $We, \lambda, \beta$) as `nn.Parameter` to be learned jointly with the physical fields via backpropagation through the scaled PDE residuals.

## Comparison: Original ViscoelasticNet vs. Current Repository Implementation

While the repository's `ViscoelasticCombinedModel` and staged training orchestration are heavily inspired by Thakur et al.'s ViscoelasticNet, there are two fundamental conceptual differences in how pressure and kinematics are handled:

### 1. Optimization Strategy: Sequential vs. Staged Co-Adaptation
- **Original ViscoelasticNet (Thakur et al.)**: Employs a strictly **sequential (decoupled)** approach. The velocity ($\phi$) and stress ($\theta$) networks are first trained using dense velocity data and constitutive equations. Once trained, **both velocity and stress networks are completely frozen**. The pressure network ($\kappa$) is then trained in complete isolation to satisfy the Navier-Stokes momentum equation, acting purely as a Poisson solver over a fixed velocity/stress field.
- **Current Repository (`train_ViscoelasticPINN`)**: Uses a **staged co-adaptation** approach:
  - *Phase 1*: Freezes pressure (`model_p`) and momentum PDEs, training `psi` and `tau` (matching Thakur et al.).
  - *Phase 2*: Freezes stress (`model_tau`), but **keeps both pressure (`model_p`) and stream function (`model_psi`) active**. This allows velocity and pressure to dynamically co-adapt to satisfy Navier-Stokes momentum conservation.
  - *Phase 3 (L-BFGS)*: Unfreezes all networks (`psi`, `p`, `tau`) for a fully coupled global refinement in FP64 (an enhancement not present in the original paper).

### 2. Pressure Boundary Conditions & Inverse Well-Posedness
- **Original ViscoelasticNet (Thakur et al.)**: In forward incompressible flows where viscosity is known, pressure appears only as a gradient ($\nabla p$) in the Navier-Stokes equations. The pressure field is defined up to an arbitrary additive constant, which the authors fix by setting a single reference point (e.g., $p(x_0) = 0$).
- **Current Repository (Inverse Problem)**: To successfully identify the unknown solvent viscosity ($\mu_s$) in channel flow (Poiseuille), knowing a single pressure anchor is mathematically insufficient; the network must know the **pressure drop ($\Delta P$)**. Therefore, the repository enforces Dirichlet boundary conditions at both the **Outlet** ($p=0$) and the **Inlet** ($p=p_{exact}$), alongside a Neumann condition at the walls ($p_y=0$). This closes the mathematical formulation, allowing the PINN to discover the true physical viscosities without diverging.

## References
- [[Thakur_et_al_ViscoelasticNet]]
- [[Oldroyd_B_Model]]
- [[Viscoelasticity]]
- [[Dynamic_Weighting]]
- [[Staged_Precision_Strategy]]
