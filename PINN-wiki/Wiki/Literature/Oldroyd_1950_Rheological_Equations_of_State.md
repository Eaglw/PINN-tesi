# On the Formulation of Rheological Equations of State (James G. Oldroyd, 1950)

## Summary
- **Author**: James G. Oldroyd (*Proceedings of the Royal Society of London. Series A*, Vol. 200, No. 1063, pp. 523–541, 1950).
- **Core Focus**: Seminal mathematical paper that established the modern principle of **Material Frame Indifference** (Objectivity) and formulated observer-invariant differential constitutive equations for non-Newtonian, viscoelastic liquids.
- **Role in Project**: The primary historical and theoretical authority for the derivation of the **Upper-Convected Time Derivative (UCTD)** and the 8-constant Oldroyd model whose reduced form constitutes the **Oldroyd-B** (or convected Jeffreys) model used across the entire PINN-tesi research.

---

## Key Methodology & Physical Principles

### 1. Principle of Material Frame Indifference (Objectivity)
- A constitutive equation must describe intrinsic material response and cannot depend on:
  1. The arbitrary spatial position or orientation of the coordinate system.
  2. The rigid-body translational motion of the observer.
  3. The time-dependent rigid-body rotational motion of the frame of reference.
- Standard partial time derivatives $\frac{\partial \boldsymbol{\tau}}{\partial t}$ and material/substantial derivatives $\frac{D\boldsymbol{\tau}}{Dt} = \frac{\partial \boldsymbol{\tau}}{\partial t} + \boldsymbol{u}\cdot\nabla\boldsymbol{\tau}$ do **not** transform objectively under time-dependent coordinate rotations.

### 2. The Upper-Convected Time Derivative (UCTD)
- To maintain invariance under arbitrary coordinate changes moving and deforming with the fluid element (codeformational frame), Oldroyd derived the upper-convected time derivative $\overset{\nabla}{\boldsymbol{\tau}}$ (often denoted $\boldsymbol{\tau}_{(1)}$ in modern notation):
  $$ \overset{\nabla}{\boldsymbol{\tau}} \equiv \frac{\partial \boldsymbol{\tau}}{\partial t} + \boldsymbol{u} \cdot \nabla \boldsymbol{\tau} - (\nabla \boldsymbol{u})^T \cdot \boldsymbol{\tau} - \boldsymbol{\tau} \cdot \nabla \boldsymbol{u} $$
- **Physical Interpretation of Terms**:
  - $\frac{\partial \boldsymbol{\tau}}{\partial t}$: Local unsteady rate of change at a fixed spatial coordinate.
  - $\boldsymbol{u} \cdot \nabla \boldsymbol{\tau}$: Convective transport of stress by the macroscopic flow.
  - $- (\nabla \boldsymbol{u})^T \cdot \boldsymbol{\tau} - \boldsymbol{\tau} \cdot \nabla \boldsymbol{u}$: Tensorial rotation and stretching of the internal stress field caused by the spatial velocity gradient $\nabla \boldsymbol{u}$.

### 3. The Oldroyd 8-Constant Model and Reduction to Oldroyd-B
- Oldroyd proposed a general differential relation involving up to 8 scalar material constants relating the extra-stress tensor $\boldsymbol{T}$ and the rate-of-strain tensor $\boldsymbol{D} = \frac{1}{2}\left[ \nabla \boldsymbol{u} + (\nabla \boldsymbol{u})^T \right]$.
- When simplified to describe a dilute suspension of non-interacting Hookean elastic dumbbells in a Newtonian solvent, the model reduces to the **Oldroyd-B** (or convected Jeffreys) constitutive equation:
  $$ \boldsymbol{T} + \lambda_1 \overset{\nabla}{\boldsymbol{T}} = 2\eta_0 \left( \boldsymbol{D} + \lambda_2 \overset{\nabla}{\boldsymbol{D}} \right) $$
  where:
  - $\lambda_1$: Relaxation time (characterizing the elastic recovery rate of polymer chains).
  - $\lambda_2$: Retardation time, related to $\lambda_1$ via the solvent viscosity ratio $\lambda_2 = \frac{\eta_s}{\eta_0}\lambda_1 = \beta \lambda_1$.
  - $\eta_0 = \eta_s + \eta_p$: Total zero-shear-rate dynamic viscosity.

### 4. Polymeric Stress Split Formulation
- In computational fluid dynamics and PINN implementations, the model is universally written in decoupled solvent-polymeric form:
  $$ \boldsymbol{T} = 2\eta_s \boldsymbol{D} + \boldsymbol{\tau} $$
  $$ \boldsymbol{\tau} + \lambda_1 \overset{\nabla}{\boldsymbol{\tau}} = 2\eta_p \boldsymbol{D} $$
  where $\boldsymbol{\tau}$ is the purely elastic polymeric extra-stress tensor.

---

## Key Findings & Project Relevance

- **Direct Theoretical Underpinning of `src/physics.py`**:
  - The PDE residual formulation in `compute_residuals()` implements the exact upper-convected derivative defined in Equation (25) of Oldroyd (1950).
- **Physical Meaning of Weissenberg Scaling**:
  - Identifies $\lambda_1$ as the characteristic timescale that scales with the shear/extension rate to yield the dimensionless Weissenberg number $Wi = \lambda_1 \dot{\gamma}_{\text{ref}}$.
- **BibTeX Citation**:
  ```bibtex
  @article{oldroyd1950formulation,
    title={On the formulation of rheological equations of state},
    author={Oldroyd, James G.},
    journal={Proceedings of the Royal Society of London. Series A},
    volume={200},
    number={1063},
    pages={523--541},
    year={1950},
    doi={10.1098/rspa.1950.0035}
  }
  ```

---

## Related Concepts
- **Topics**: [[Viscoelasticity]], [[Upper-convected time derivative]], [[Fluid_Dynamics]], [[Nondimensionalization]]
- **Methods**: [[ViscoelasticNet]], [[ViscoelasticNet_Full model]], [[Log_Conformation_Tensor]]
- **Systems**: [[Viscoelastic_Fluids]], [[Viscoelastic_Training]], [[Thesis_Chapter_02_Fluid_Dynamics_Guide]]
