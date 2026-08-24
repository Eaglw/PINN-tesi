# Method: ViscoelasticNet

## Overview
ViscoelasticNet is a deep learning framework designed to solve both forward and inverse problems in viscoelastic fluid mechanics. It extends the standard PINN approach by incorporating constitutive laws directly into the loss function.

## Technical Implementation
Key features of the current repository implementation:
- **Stream Function Formulation**: Velocity is derived from a stream function network ($u = \psi_y, v = -\psi_x$), ensuring divergence-free flow ($\nabla \cdot \mathbf{u} = 0$) by construction.
- **Multi-Network Architecture**: Uses `ViscoelasticCombinedModel` to unify separate networks. This architectural separation leverages the physical [[Pressure_Stress_Decoupling]] between pressure and extra-stress:
    - **Stream Function ($\psi$)**: Scalar output.
    - **Pressure ($p$)**: Scalar output.
    - **Stress ($\tau$)**: 3-output network for $\tau_{xx}, \tau_{xy}, \tau_{yy}$.
- **Physical Loss**:
    - **Momentum**: Couples velocity, pressure, and the divergence of the extra stress tensor.
    - **Oldroyd-B**: Implements the upper-convected constitutive equation.
- **Staged Training (Decoupling)**: Implements a 2-stage decoupled training strategy (Phase 1: Kinematics & Rheology $\to$ Phase 2: Hydrodynamics & Pressure) to stabilize convergence. Note that Phase 3 (fully coupled joint optimization) is formally deprecated due to numerical destabilization of stress fields. See [[Staged_Training_Procedure]] for details.
- **Precision Switching**: Transitions from Adam (exploration) to L-BFGS (refinement) while switching from `float32` to `float64`. See [[Staged_Precision_Strategy]].

## Recent Updates: Semi-Inverse Strategy (Goal 1)
To fully align with the original ViscoelasticNet methodology, a `semi_inverse` mode (Goal 1) has been implemented:
- **Supervision on Velocity**: The internal data loss is computed strictly as $MSE(u_{pred}, u_{obs}) + MSE(v_{pred}, v_{obs})$, driving the stream function $\psi$ purely via its derivatives. No internal stress data from CFD is ever fed to the PINN.
- **Variance Scaling**: All loss components (PDE, BC, Data) are normalized by the variance of the reference velocity field ($max(\sigma^2_u, 1e-8)$) to balance gradients.
- **Optimization Enhancements**: Utilizes mini-batching ($N_{int}=256, N_{bc}=64$) to stochasticize the gradient descent, coupled with a Cosine Annealing learning rate scheduler and an Adam `eps=1e-7` for robust exploration.

### Future Extension: `inverse_dense` (Full-Field Inverse Problem)
To extend the framework to a fully inverse scenario ("inverse_dense") matching ViscoelasticNet with dense PIV/CFD data:
1. **Full-Field Collocation**: Replace the sparse, randomly sampled collocation points with the dense, uniform PIV/CFD grid ($xy\_grid\_flat$).
2. **Global Loss Calculation**: Compute the data loss across the entire spatial domain rather than a subset.
3. **Unknown Parameter Identification**: Define physical parameters (e.g., $We, \lambda, \beta$) as `nn.Parameter` to be learned jointly with the physical fields via backpropagation through the scaled PDE residuals.

## Comparison: Original ViscoelasticNet vs. Current Repository Implementation

While the repository's `CombinedModel` and staged training orchestration are heavily inspired by Thakur et al.'s ViscoelasticNet, there are two key evolutionary aspects in how pressure, kinematics, and geometry are handled:

### 1. Optimization Strategy: Decoupled 2-Stage Pipeline
- **Original ViscoelasticNet (Thakur et al.)**: Employs a strictly sequential decoupled approach. The velocity ($\phi$) and stress ($\theta$) networks are first trained using dense velocity data and constitutive equations. Once trained, both velocity and stress networks are completely frozen. The pressure network ($\kappa$) is then trained in complete isolation to satisfy the Navier-Stokes momentum equation, acting purely as a Poisson solver over a fixed velocity/stress field.
- **Current Repository (`final_roll`)**: Implements a robust **Decoupled 2-Phase Staged Training**:
  - *Phase 1 (Kinematics & Rheology)*: Freezes pressure (`model_p`) and deactivates momentum ($w_{mom}=0$). Trains stream function (`model_psi`) and extra-stress (`model_tau`) with Adam FP32 followed by L-BFGS FP64 to discover $\lambda$ and topological stress distribution.
  - *Phase 2 (Hydrodynamics & Pressure)*: Freezes both `model_psi` and `model_tau`, precomputes the divergence of extra-stress $\nabla \cdot \boldsymbol{\tau}$ as a static tensor to eliminate autograd graph overhead, and trains `model_p` with active momentum ($w_{mom}=1.0$) with Adam FP32 + L-BFGS FP64 to discover solvent viscosity $\beta$ / $\mu_s$.
  - *Deprecation of Phase 3*: A fully coupled Phase 3 (unfreezing all networks simultaneously) was tested but found to be destructive: backpropagating momentum residuals into $\boldsymbol{\tau}$ corrupts the learned relaxation time $\lambda$ and induces stress artifacts.

### 2. Boundary Conditions & Geometry: Channel Flow vs. Four-Roll Mill
- **Poiseuille Channel Flow (Historical Benchmark)**: In open channel flow, identifying solvent viscosity $\mu_s$ requires prescribing the total pressure drop ($\Delta P$) across inlet and outlet to break the scale degeneracy (see [[Analisi geometria in tubo semplice]]).
- **Four-Roll Mill (Current Geometry)**: In the closed 4-roll mill domain, pressure is anchored at a single spatial point ($p(x_0, y_0) = 0$), wall no-slip BCs are enforced on external boundaries, rotating roll velocities are enforced on cylinder surfaces, and extra-stress BCs on roll surfaces provide the necessary scale anchor for $\tau$.

## References
- [[Thakur_et_al_ViscoelasticNet]]
- [[Oldroyd_B_Model]]
- [[Viscoelasticity]]
- [[Dynamic_Weighting]]
- [[Staged_Precision_Strategy]]
- [[Pressure_Stress_Decoupling]]
