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
- **Full-Grid Deterministic Collocation**: Rather than sparse, stochastic mini-batching, the solver evaluates the loss deterministically over the entire dense FEM/CFD spatial mesh (12k, 29k, 52k, 125k nodes) using sequential autograd chunking to cap peak VRAM.
- **Adam & Cosine Annealing**: Powered by synchronized Cosine Annealing ($2.5 \times 10^{-3} \to 2.5 \times 10^{-6}$) and differentiated Adam epsilon ($\epsilon_{net} = 10^{-8}, \epsilon_{phys} = 10^{-15}$).

## Comparison: Original ViscoelasticNet vs. Current Repository Implementation

While the repository's `CombinedModel` and staged training orchestration are heavily inspired by Thakur et al.'s ViscoelasticNet, there are two key evolutionary aspects in how pressure, kinematics, and geometry are handled:

### 1. Optimization Strategy: Decoupled 2-Stage Pipeline
- **Original ViscoelasticNet (Thakur et al.)**: Employs a strictly sequential decoupled approach. The velocity ($\phi$) and stress ($\theta$) networks are first trained using dense velocity data and constitutive equations. Once trained, both velocity and stress networks are completely frozen. The pressure network ($\kappa$) is then trained in complete isolation to satisfy the Navier-Stokes momentum equation, acting purely as a Poisson solver over a fixed velocity/stress field.
- **Current Repository (`final_roll`)**: Implements a robust **Decoupled 2-Phase Staged Training**:
  - *Phase 1 (Kinematics & Rheology)*: Freezes pressure (`model_p`) and deactivates momentum ($w_{mom}=0$). Trains stream function (`model_psi`) and extra-stress (`model_tau`) with Adam FP32 followed by L-BFGS FP64 to discover $\lambda, \mu_p$, and non-linear parameters $\alpha, \varepsilon$.
  - *Phase 2 (Hydrodynamics & Pressure)*: Freezes **only `model_tau`** (extra-stress), precomputes its divergence $\nabla \cdot \boldsymbol{\tau}$ as a static tensor to eliminate autograd graph overhead, and trains `model_p` with active momentum ($w_{mom}=1.0$). **Crucially, `model_psi` is NEVER frozen in Phase 2**: it remains mobile with a controlled micro-learning rate ($LR_\psi \approx 10^{-4}$) and [[Soft_Anti_Drift]] regularization to compensate for the irrotational velocity error, overcoming the [[Pressure_Stress_Decoupling#The Helmholtz-Hodge Pressure Inference Limit|Helmholtz-Hodge limit]] and recovering the true pressure gradient $\nabla p$.
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
