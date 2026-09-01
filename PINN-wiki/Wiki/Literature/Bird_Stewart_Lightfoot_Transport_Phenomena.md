# Transport Phenomena (Bird, Stewart, Lightfoot)

## Summary
- **Authors**: R. Byron Bird, Warren E. Stewart, Edwin N. Lightfoot (University of Wisconsin-Madison, 2nd Edition / Revised).
- **Core Focus**: The landmark foundational textbook establishing the unified continuum framework for momentum, energy, and mass transport.
- **Role in Project**: Serves as the primary theoretical reference for conservation laws (continuity and Navier-Stokes equations), shell momentum balances, Cauchy stress tensor formulation, and classical laminar flow analytical solutions.

---

## Key Methodology & Physical Principles

### 1. Conservation Laws for Incompressible Fluids
- **Mass Conservation (Continuity Equation)**:
  $$ \nabla \cdot \mathbf{u} = 0 \quad \iff \quad \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0 $$
- **Momentum Conservation (Cauchy Momentum Balance)**:
  $$ \rho \left( \frac{\partial \mathbf{u}}{\partial t} + \mathbf{u} \cdot \nabla \mathbf{u} \right) = -\nabla p - \nabla \cdot \boldsymbol{\tau} + \rho \mathbf{g} $$
  where $\rho$ is density, $p$ is hydrostatic pressure, and $\boldsymbol{\tau}$ is the viscous/extra-stress tensor.

### 2. Viscous Stress Tensor & Newtonian Fluid
- For Newtonian fluids with constant viscosity $\mu$:
  $$ \boldsymbol{\tau} = -\mu \left( \nabla \mathbf{u} + (\nabla \mathbf{u})^T \right) = -2\mu \mathbf{D} $$
- Substituting into the Cauchy momentum balance yields the classical **Navier-Stokes Equation**:
  $$ \rho \frac{D\mathbf{u}}{Dt} = -\nabla p + \mu \nabla^2 \mathbf{u} + \rho \mathbf{g} $$

### 3. Shell Momentum Balances & Benchmark Flows
- **Planar Poiseuille Flow**: Laminar flow between parallel plates driven by a constant pressure gradient $\frac{dp}{dx} = -G$:
  $$ u(y) = \frac{G}{2\mu} y (H - y) = u_{\max} \left[ 1 - \left( \frac{2y - H}{H} \right)^2 \right] $$
- **Couette Flow**: Flow driven by boundary motion (drag flow).
- **Non-Newtonian Generalized Fluids**: Introduction of shear-rate dependent viscosity $\eta(\dot{\gamma})$ (Power-law, Carreau-Yasuda, Bingham plastics).

---

## Key Findings & Project Relevance

- **Foundational Formulation of Project PDE Residuals**:
  - Direct source for the conservation equations encoded in `physics.py` and evaluated via Autograd.
  - Reference benchmark for Poiseuille dataset generation and boundary condition validation.
- **Stress-Tensor Scomposition**:
  - Rigorous definition of the total Cauchy stress tensor $\boldsymbol{\sigma} = -p\mathbf{I} - \boldsymbol{\tau}$, which provides the theoretical underpinning for separating pressure (Lagrange multiplier for incompressibility) from the extra-stress tensor.

---

## Related Concepts
- **Topics**: [[Fluid_Dynamics]], [[PINN_Fundamentals]], [[Pressure_Stress_Decoupling]], [[Nondimensionalization]]
- **Methods**: [[ViscoelasticNet]], [[Viscoelastic_Residual_Scaling]], [[COMSOL_Boundary_Extraction]]
- **Systems**: [[Viscoelastic_Fluids]], [[Analisi geometria in tubo semplice]], [[Heat2D_Analysis]]
