# Dynamics of Polymer Liquids, Vol. 1: Fluid Mechanics (Bird, Armstrong, Hassager)

## Summary
- **Authors**: R. Byron Bird, Robert C. Armstrong, Ole Hassager (Wiley-Interscience, 2nd Edition, 1987).
- **Core Focus**: The universally recognized standard treatise on the fluid mechanics, rheology, and constitutive modeling of polymeric liquids and non-Newtonian materials.
- **Role in Project**: The authoritative theoretical reference for all viscoelastic constitutive models (Upper-Convected Maxwell, Oldroyd-B, Phan-Thien–Tanner, Giesekus), frame-indifferent tensor rates (Upper-Convected Derivative), normal stress phenomena, and viscoelastic dimensionless groups ($Wi, De, \beta$).

---

## Key Methodology & Physical Principles

### 1. Rheological Phenomena & Material Functions
- **Non-Newtonian Behaviors**: Shear-thinning viscosity $\eta(\dot{\gamma})$, stress relaxation upon cessation of flow, creep and elastic recovery.
- **Normal Stress Differences**:
  $$ N_1(\dot{\gamma}) = \tau_{xx} - \tau_{yy}, \quad N_2(\dot{\gamma}) = \tau_{yy} - \tau_{zz} $$
  responsible for elastic effects like rod-climbing (Weissenberg effect), extrudate swell (die swell), and vortex enhancement in contractions.

### 2. Objective Time Derivatives (Principle of Material Frame Indifference)
- Standard partial $\frac{\partial}{\partial t}$ or material $\frac{D}{Dt}$ derivatives of second-order tensors are not invariant under time-dependent coordinate frame rotations.
- **Upper-Convected (Oldroyd) Time Derivative** $\boldsymbol{\tau}_{(1)}$:
  $$ \boldsymbol{\tau}_{(1)} = \frac{D\boldsymbol{\tau}}{Dt} - (\nabla \mathbf{u})^T \cdot \boldsymbol{\tau} - \boldsymbol{\tau} \cdot \nabla \mathbf{u} $$
  where $\frac{D\boldsymbol{\tau}}{Dt} = \frac{\partial \boldsymbol{\tau}}{\partial t} + \mathbf{u} \cdot \nabla \boldsymbol{\tau}$.
- In 2D Cartesian coordinates for steady flow ($u = \partial_y \psi, v = -\partial_x \psi$):
  $$ (\mathbf{u} \cdot \nabla \boldsymbol{\tau})_{ij} = u \frac{\partial \tau_{ij}}{\partial x} + v \frac{\partial \tau_{ij}}{\partial y} $$
  $$ [(\nabla \mathbf{u})^T \cdot \boldsymbol{\tau} + \boldsymbol{\tau} \cdot \nabla \mathbf{u}] = \begin{pmatrix} 2\frac{\partial u}{\partial x}\tau_{xx} + 2\frac{\partial u}{\partial y}\tau_{xy} & \frac{\partial v}{\partial x}\tau_{xx} + \left(\frac{\partial u}{\partial x}+\frac{\partial v}{\partial y}\right)\tau_{xy} + \frac{\partial u}{\partial y}\tau_{yy} \\ \text{sym} & 2\frac{\partial v}{\partial x}\tau_{xy} + 2\frac{\partial v}{\partial y}\tau_{yy} \end{pmatrix} $$

### 3. Differential Constitutive Equations

#### A. Upper-Convected Maxwell (UCM) Model
$$ \boldsymbol{\tau} + \lambda \boldsymbol{\tau}_{(1)} = -2\eta_0 \mathbf{D} $$

#### B. Oldroyd-B Model (Solvent + Polymer Stress Split)
- Decomposes total extra-stress into solvent Newtonian stress $\boldsymbol{\tau}_s$ and polymer viscoelastic stress $\boldsymbol{\tau}_p$:
  $$ \boldsymbol{\tau} = \boldsymbol{\tau}_s + \boldsymbol{\tau}_p, \quad \boldsymbol{\tau}_s = -2\eta_s \mathbf{D} $$
  $$ \boldsymbol{\tau}_p + \lambda \boldsymbol{\tau}_{p(1)} = -2\eta_p \mathbf{D} $$
- Retardation time $\lambda_2 = \frac{\eta_s}{\eta_s + \eta_p}\lambda = \beta \lambda$.

#### C. Giesekus Model (Molecular Anisotropy)
- Adds a quadratic stress term modeling anisotropic hydrodynamic drag on polymer chains ($\alpha \in [0, 0.5]$):
  $$ \boldsymbol{\tau}_p + \lambda \boldsymbol{\tau}_{p(1)} + \frac{\alpha \lambda}{\eta_p} (\boldsymbol{\tau}_p \cdot \boldsymbol{\tau}_p) = -2\eta_p \mathbf{D} $$

#### D. Phan-Thien–Tanner (PTT) Model (Network Destruction)
- Based on transient network theory with rate of junction creation/destruction dependent on trace of stress:
  $$ f(\text{tr}\,\boldsymbol{\tau}_p)\boldsymbol{\tau}_p + \lambda \boldsymbol{\tau}_{p(1)} = -2\eta_p \mathbf{D}, \quad f(\text{tr}\,\boldsymbol{\tau}_p) = 1 + \frac{\epsilon \lambda}{\eta_p} \text{tr}(\boldsymbol{\tau}_p) $$

### 4. Dimensionless Groups
- **Weissenberg Number ($Wi$)**: Ratio of elastic to viscous forces ($Wi = \lambda \dot{\gamma}_{\text{ref}} = \lambda \frac{U}{H}$).
- **Deborah Number ($De$)**: Ratio of relaxation time to process observation time ($De = \frac{\lambda}{t_{\text{flow}}}$).
- **Viscosity Ratio ($\beta$)**: Solvent fraction $\beta = \frac{\eta_s}{\eta_s + \eta_p} \in [0, 1]$.

---

## Key Findings & Project Relevance

- **Direct Foundation of Viscoelastic Constitutive Losses**:
  - The unified constitutive equation in `src/physics.py` and `ViscoelasticNet_Full model.md` is derived directly from chapters 7 and 8 of BAH Vol. 1.
- **Analytical Reference for Normal Stresses & Inversion**:
  - Provides the fundamental theoretical basis for the parameter identifiability study: $\lambda$ governs normal stress growth ($N_1 = 2\lambda \eta_p \dot{\gamma}^2$), while $\eta_p$ scales shear stress ($\tau_{xy} = \eta_p \dot{\gamma}$).
- **Log-Space Parametrization**:
  - Enforces physical positivity of relaxation time ($\lambda > 0$) and polymer viscosity ($\eta_p > 0$) consistent with thermodynamic admissibility.

---

## Related Concepts
- **Topics**: [[Viscoelasticity]], [[Upper-convected time derivative]], [[Fluid_Dynamics]], [[Viscoelastic_Parameter_Identifiability]], [[Nondimensionalization]]
- **Methods**: [[ViscoelasticNet]], [[ViscoelasticNet_Full model]], [[Lasso_Regularization]], [[Log_Conformation_Tensor]]
- **Systems**: [[Viscoelastic_Fluids]], [[Viscoelastic_Training]], [[Analisi geometria in tubo semplice]]
