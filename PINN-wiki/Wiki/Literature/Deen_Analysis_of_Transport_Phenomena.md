# Analysis of Transport Phenomena (William M. Deen)

## Summary
- **Author**: William M. Deen (Massachusetts Institute of Technology - MIT, Oxford University Press, 2nd Edition).
- **Core Focus**: Advanced, mathematically rigorous analysis of momentum, heat, and mass transport using vector and tensor calculus, perturbation theory, asymptotic scaling, and stream function/vorticity formulations.
- **Role in Project**: Primary authority on scaling analysis, dimensional reduction, the mathematical derivation of the **Stream Function** ($\psi$) for incompressible 2D flows, and the **Vorticity Transport Equation** ($\omega$) obtained by taking the curl of momentum.

---

## Key Methodology & Physical Principles

### 1. Vector & Tensor Mechanics of Continuous Media
- Formalization of coordinate-free differential operators (gradient $\nabla$, divergence $\nabla \cdot$, curl $\nabla \times$, Laplacian $\nabla^2$) and curvilinear coordinate mappings.
- Exact kinematics: Deformation rate tensor $\mathbf{D} = \frac{1}{2}\left( \nabla \mathbf{u} + (\nabla \mathbf{u})^T \right)$ and spin/vorticity tensor $\mathbf{W} = \frac{1}{2}\left( \nabla \mathbf{u} - (\nabla \mathbf{u})^T \right)$.

### 2. Stream Function Formulation for 2D Incompressible Flows
- Mass conservation $\nabla \cdot \mathbf{u} = \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0$ is satisfied identically by defining a scalar stream function $\psi(x, y)$:
  $$ u = \frac{\partial \psi}{\partial y}, \quad v = -\frac{\partial \psi}{\partial x} $$
- **Significance for Neural Networks**: Eliminates continuity as a soft loss penalty, embedding exact mass conservation into the network architecture by construction.

### 3. Vorticity Transport Equation & Curl of Momentum
- Taking the curl ($\nabla \times$) of the Cauchy momentum equation eliminates the pressure gradient field $\nabla p$ identically, because $\nabla \times (\nabla p) \equiv \mathbf{0}$.
- For 2D incompressible Newtonian/viscoelastic flow:
  $$ \omega_z = (\nabla \times \mathbf{u})_z = \frac{\partial v}{\partial x} - \frac{\partial u}{\partial y} = -\nabla^2 \psi $$
  $$ \rho \left( \frac{\partial \omega_z}{\partial t} + \mathbf{u} \cdot \nabla \omega_z \right) = \mu_s \nabla^2 \omega_z + \left[ \nabla \times (\nabla \cdot \boldsymbol{\tau}_p) \right]_z $$
- **Compatibility Condition**: $\nabla p = \mathbf{F}(\mathbf{u}, \boldsymbol{\tau}) \implies \nabla \times \mathbf{F} = \mathbf{0}$.

### 4. Rigorous Nondimensionalization & Scaling Analysis
- Identification of characteristic length $H$, velocity $U$, time $H/U$, and viscous stress scales $\eta_0 U/H$.
- Formulation of the dimensionless Reynolds number $Re = \frac{\rho U H}{\eta_0}$ and Stokes/creeping flow limits ($Re \to 0 \implies \nabla^4 \psi = 0$).

---

## Key Findings & Project Relevance

- **Architectural Backbone of `CombinedModel`**:
  - The stream function formulation $\psi \to (u, v)$ is the core inductive bias of our PINN, ensuring zero mass-residual error across all training phases.
- **Vorticity Inversion & Decoupled Solvent Identification**:
  - Direct theoretical justification for eliminating pressure indeterminacy in Phase 2 using the vorticity transport equation and the rotational compatibility condition $\text{curl}(\mathbf{F}) = 0$.
- **Adimensionalization Protocols**:
  - Guiding methodology for decoupled scaling ($Re_{\text{scale}}$ vs. physical parameters) and viscous stress normalization.

---

## Related Concepts
- **Topics**: [[Fluid_Dynamics]], [[PINN_Fundamentals]], [[Nondimensionalization]], [[Pressure_Stress_Decoupling]]
- **Methods**: [[Vorticity_Inversion_Solvent]], [[Vorticity_Regularization]], [[Zero_Stress_BC_Compatibility]], [[Soft_Anti_Drift]]
- **Systems**: [[Viscoelastic_Fluids]], [[Viscoelastic_Training]], [[Heat2D_Analysis]]
