# Computational Rheology (Robert G. Owens, Timothy N. Phillips)

## Summary
- **Authors**: Robert G. Owens, Timothy N. Phillips (Imperial College Press / World Scientific, 2002).
- **Core Focus**: Comprehensive treatise on numerical and computational methods for simulating non-Newtonian, polymeric, and viscoelastic fluid flows.
- **Role in Project**: Primary authority on the mathematical classification of viscoelastic systems (mixed elliptic-hyperbolic character), numerical instability mechanisms (**High Weissenberg Number Problem - HWNP**), stress stabilization formulations (EVSS, DEVSS, Log-conformation), and standard benchmark geometries (Four-roll mill, cross-slot, planar contraction, cylinder in channel).

---

## Key Methodology & Numerical Challenges

### 1. Mixed Elliptic-Hyperbolic Mathematical Classification
- **Elliptic Nature**: Mass conservation ($\nabla \cdot \mathbf{u} = 0$) and momentum diffusion ($\mu_s \nabla^2 \mathbf{u} - \nabla p$) exhibit elliptic spatial character, propagating information across the entire domain.
- **Hyperbolic Nature**: The constitutive transport equation ($\mathbf{u} \cdot \nabla \boldsymbol{\tau} - (\nabla \mathbf{u})^T \cdot \boldsymbol{\tau} - \boldsymbol{\tau} \cdot \nabla \mathbf{u} + \frac{1}{\lambda}\boldsymbol{\tau} = \dots$) is hyperbolic, transporting elastic stress strictly along fluid streamlines (characteristics).
- **Consequence for PINNs**: Loss functions combine elliptic boundary-value constraints with hyperbolic wave-like advection, requiring decoupled optimization strategies (e.g., Staged Training).

### 2. The High Weissenberg Number Problem (HWNP)
- Above critical values of $Wi$ ($Wi \gtrsim 0.5 - 1.0$ depending on geometry), standard numerical methods (FEM, FVM, spectral) experience catastrophic loss of convergence.
- **Root Causes**:
  - Exponential growth of normal stresses ($\tau_{xx}, \tau_{yy}$) in regions of high shear and stagnation points.
  - Formation of ultra-steep boundary layers and localized stress cusps that standard polynomial bases cannot resolve without losing positive-definiteness of the conformation tensor $\mathbf{A}$.

### 3. Stabilization Techniques & Conformation Tensor
- **Conformation Tensor Formulation**:
  $$ \mathbf{A} = \mathbf{I} + \frac{\lambda}{\eta_p} \boldsymbol{\tau}_p $$
  Thermodynamic admissibility requires $\mathbf{A}$ to be strictly positive-definite ($\det(\mathbf{A}) > 0$).
- **Log-Conformation Representation**:
  Setting $\mathbf{s} = \log \mathbf{A}$ transforms the exponential stress growth into a linear equation for $\mathbf{s}$, automatically preserving positive-definiteness and eliminating the HWNP barrier.
- **EVSS / DEVSS (Elastic-Viscous Split Stress)**:
  Extracts an explicit elliptic Laplacian term from the constitutive equation to stabilize the momentum solver.

### 4. The Four-Roll Mill Benchmark Geometry
- Consists of four counter-rotating circular cylinders placed symmetrically in a square/cylindrical chamber.
- **Flow Kinematics**:
  - Produces a central **stagnation point** at $(0,0)$ surrounded by a region of nearly pure planar extensional flow ($\mathbf{u} \approx (\dot{\epsilon}x, -\dot{\epsilon}y)$).
  - High velocity gradients near cylinder surfaces $\Gamma_{\text{rolls}}$ coupled with strong extensional stretching along the central axes ($x = 0, y = 0$).
- **Birefringence & Stress Localization**: High localized stresses accumulate along the outflow axis, serving as the quintessential testbed for viscoelastic parameter estimation and stress prediction.

---

## Key Findings & Project Relevance

- **Direct Blueprint for the 4-Roll Mill Setup**:
  - The geometry, boundary conditions, and flow regimes in `final_roll/` directly implement the four-roll mill benchmark analyzed by Owens & Phillips.
- **Physical Explanation of Training Bottlenecks**:
  - Explains why standard unified training struggles: the mixed elliptic-hyperbolic nature causes gradient pathologies between elliptic pressure/viscous diffusion and hyperbolic stress advection.
  - Justifies our **Staged Training Framework**: solving kinematics and stress along streamlines in Phase 1 before solving global elliptic pressure in Phase 2.
- **Support for Advanced Regularizations**:
  - Motivates the use of log-conformation transformations, vorticity regularization, and soft anti-drift to overcome stress singularities near roll surfaces.

---

## Related Concepts
- **Topics**: [[Viscoelasticity]], [[Fluid_Dynamics]], [[Pressure_Stress_Decoupling]], [[Spectral_Bias]], [[Nondimensionalization]]
- **Methods**: [[Log_Conformation_Tensor]], [[Staged_Training_Procedure]], [[ViscoelasticNet]], [[Viscoelastic_Residual_Scaling]], [[Vorticity_Regularization]]
- **Systems**: [[Viscoelastic_Fluids]], [[Viscoelastic_Training]]
