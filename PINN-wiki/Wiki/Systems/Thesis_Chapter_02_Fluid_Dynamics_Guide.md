# Thesis Chapter 02: Viscoelastic Fluid Mechanics — Writing & Literature Guide

## Summary
- **Purpose**: Definitive, exhaustive study and formulation guide for writing and rewriting **Chapter 2 (Viscoelastic Fluid Mechanics)** of the master's thesis.
- **Core Strategy**: Unifies classical continuum fluid mechanics (**Deen**, **BSL**), molecular polymer rheology (**Bird, Armstrong, Hassager - DPL Vol. 1**, **Oldroyd 1950**), and computational/numerical rheology (**Owens & Phillips**) with direct links to the PINN implementation in `final_roll/src/physics.py`.
- **Target Audience**: Authoritative reference for LaTeX drafting, mathematical derivations, equation cross-referencing, and BibTeX fact-checking.

---

## 1. Master Literature Reading List by Book & Chapter

### A. William M. Deen – *Analysis of Transport Phenomena* (2nd Edition, Oxford Univ. Press)
*Primary authority for continuum mechanics, asymptotic scaling, dimensionless groups, and stream function derivations.*
- **Capitolo 1 (Continuum Modeling)**:
  - *Sec. 1.1–1.6*: Continuum hypothesis validity for single-phase polymer solutions, characteristic microscopic vs. macroscopic length/time scales.
- **Capitolo 2 & Capitolo 6 (Conservation of Mass and Momentum)**:
  - *Sec. 6.1*: Control volume momentum balances, differential Cauchy momentum equation $\rho \frac{D\boldsymbol{u}}{Dt} = \nabla \cdot \boldsymbol{\sigma} + \rho \boldsymbol{g}$, decomposition of Cauchy stress tensor into thermodynamic pressure $p$ and extra-stress $\boldsymbol{T}$.
  - *Sec. 6.4*: Exact fluid kinematics: decomposition of $\nabla \boldsymbol{u}$ into symmetric rate-of-strain tensor $\boldsymbol{D} = \frac{1}{2}(\nabla \boldsymbol{u} + (\nabla \boldsymbol{u})^T)$ and anti-symmetric spin/vorticity tensor $\boldsymbol{W} = \frac{1}{2}(\nabla \boldsymbol{u} - (\nabla \boldsymbol{u})^T)$.
  - *Sec. 6.5 & 6.7*: Solid boundary conditions (no-slip) and surface stress integration.
- **Capitolo 3 (Scaling, Order of Magnitude, and Dimensionless Numbers)**:
  - *Sec. 3.1–3.4*: Rigorous non-dimensionalization and scaling analysis.
  - Definition and derivation of the **Bond number** $Bo = \frac{\Delta\rho \, g \, L^2}{\gamma} \ll 1$ justifying negligible gravity body forces in micro-scale devices.
  - Derivation of the **Reynolds number** $Re = \frac{\rho U H}{\mu_0}$.
- **Capitolo 7 (Viscous Flow & Stream Function Formulation) — Core Inductive Bias**:
  - *Sec. 7.1–7.3*: Derivation of the **Stream Function** $\psi(x, y)$ for 2D planar incompressible flow:
    $$ u = \frac{\partial \psi}{\partial y}, \quad v = -\frac{\partial \psi}{\partial x} \implies \nabla \cdot \boldsymbol{u} = \frac{\partial^2 \psi}{\partial x \partial y} - \frac{\partial^2 \psi}{\partial y \partial x} \equiv 0 $$
  - Vorticity transport equation $\omega_z = -\nabla^2 \psi$ and pressure elimination via momentum curl ($\nabla \times \nabla p \equiv \boldsymbol{0}$).

---

### B. R. Byron Bird, Robert C. Armstrong, Ole Hassager – *Dynamics of Polymeric Liquids, Vol. 1: Fluid Mechanics* (2nd Edition, Wiley)
*The fundamental treatise on non-Newtonian polymer rheology, objective time derivatives, and differential constitutive models.*
- **Capitolo 1 (Flow Phenomena in Polymeric Liquids)**:
  - Non-Newtonian macro-phenomena: shear-thinning viscosity, rod-climbing (Weissenberg effect), die swell, elastic recoil, and normal stress generation.
- **Capitolo 3 (Material Functions for Polymeric Liquids)**:
  - *Sec. 3.1–3.4*: Steady shear flow material functions: non-Newtonian viscosity $\eta(\dot{\gamma})$, first and second normal stress difference coefficients $\Psi_1(\dot{\gamma}) = \frac{\tau_{xx}-\tau_{yy}}{\dot{\gamma}^2}$, $\Psi_2(\dot{\gamma}) = \frac{\tau_{yy}-\tau_{zz}}{\dot{\gamma}^2}$.
  - *Sec. 3.5*: Uniaxial and planar extensional flow material functions ($\eta_E(\dot{\epsilon})$), key for four-roll mill stagnation dynamics.
- **Capitolo 7 (The Convected Derivative and Frame Indifference) — ESSENTIAL**:
  - *Sec. 7.1–7.3*: Mathematical proof that partial $\frac{\partial}{\partial t}$ and material $\frac{D}{Dt}$ derivatives of second-order tensors violate material frame indifference under time-dependent observer rotations.
  - Definition of the **Upper-Convected Time Derivative (UCTD)**:
    $$ \overset{\nabla}{\boldsymbol{\tau}} = \frac{D\boldsymbol{\tau}}{Dt} - (\nabla \boldsymbol{u})^T \cdot \boldsymbol{\tau} - \boldsymbol{\tau} \cdot \nabla \boldsymbol{u} $$
- **Capitolo 8 (Differential Constitutive Equations) — ESSENTIAL**:
  - *Sec. 8.1–8.2*: Kinetic derivation of **Upper-Convected Maxwell (UCM)** and **Oldroyd-B** models from the Hookean elastic dumbbell (two friction beads connected by an entropic linear spring).
  - Polymeric stress split $\boldsymbol{T} = 2\eta_s \boldsymbol{D} + \boldsymbol{\tau}$ and relationship between relaxation time $\lambda_1$ and retardation time $\lambda_2 = \beta \lambda_1$.
  - Exact proof of the **extensional viscosity singularity** at critical extensional Weissenberg number $Wi_{\text{ext}} = 0.5$ (unbounded dumbbell elongation).
  - *Sec. 8.3*: Regularized non-linear constitutive models:
    - **Phan-Thien–Tanner (PTT)**: Stress-dependent destruction function $f(\text{tr}\,\boldsymbol{\tau}) = 1 + \frac{\varepsilon \lambda}{\eta_p}\text{tr}(\boldsymbol{\tau})$.
    - **Giesekus**: Quadratic stress term $\frac{\alpha \lambda}{\eta_p}(\boldsymbol{\tau}\cdot\boldsymbol{\tau})$ representing molecular hydrodynamic anisotropy.

---

### C. Robert G. Owens, Timothy N. Phillips – *Computational Rheology* (Imperial College Press, 2002)
*Primary authority for the CFD perspective, mixed differential systems, HWNP numerical breakdown, and boundary conditions.*
- **Capitolo 1 (Introduction)**:
  - *Sec. 1.1–1.3*: Classical viscoelastic models and early computational simulations; introduction to the **High Weissenberg Number Problem (HWNP)**.
- **Capitolo 2 (Fundamentals)**:
  - *Sec. 2.1–2.2*: Derivation of integral and differential conservation laws; symmetry and existence of the stress tensor.
  - *Sec. 2.6.1*: Explicit differential constitutive models and solvent/polymer stress splitting.
- **Capitolo 3 (Mathematical Theory of Viscoelastic Fluids)**:
  - *Sec. 3.3*: **Mixed Elliptic-Hyperbolic character**: Momentum/continuity are elliptic (global pressure and viscous diffusion), while constitutive transport is hyperbolic (convection of elastic stress along streamlines characteristics). *Direct theoretical justification for PINN Staged Training!*
  - *Sec. 3.4*: Boundary conditions for velocity and extra-stress on solid walls and inflows.
- **Capitolo 7 (Defeating the High Weissenberg Number Problem)**:
  - *Sec. 7.1*: Physical and numerical origin of HWNP (exponential growth of stress gradients near stagnation points, loss of positive-definiteness of conformation tensor $\boldsymbol{A} = \boldsymbol{I} + \frac{\lambda}{\eta_p}\boldsymbol{\tau}$).
- **Capitolo 9 (Benchmark Problems)**:
  - Stagnation point kinematics and four-roll mill benchmark flow characteristics.

---

### D. James G. Oldroyd (1950) – Seminal Paper
- **Reference**: *Proc. R. Soc. Lond. A*, 200(1063), 523–541.
- **Key Concepts**:
  - Principle of Objectivity (invariance under arbitrary frame transformations).
  - Derivation of the convected time derivatives using convected coordinate frames.
  - 8-constant differential constitutive framework and the specific formulation of Oldroyd-B:
    $$ \boldsymbol{T} + \lambda_1 \overset{\nabla}{\boldsymbol{T}} = 2\eta_0 \left( \boldsymbol{D} + \lambda_2 \overset{\nabla}{\boldsymbol{D}} \right) $$

---

## 2. Section-by-Section Theoretical Structure for Chapter 2

```
Chapter 2: Viscoelastic Fluid Mechanics
│
├── 2.1 General Conservation Laws
│   ├── Continuity Equation (Mass balance, volumetric dilation rate)
│   ├── Cauchy Momentum Equation (Material acceleration, Eulerian form)
│   └── Cauchy Stress Decomposition (Thermodynamic pressure p vs. extra-stress T)
│
├── 2.2 Physical Assumptions and System Reduction
│   ├── Single-phase continuum, incompressibility (\nabla \cdot u = 0)
│   ├── Bond number scaling (Bo << 1) and negligible gravity
│   ├── 2D steady planar flow kinematics
│   └── Polymeric stress decomposition (T = 2\mu_s D + \tau)
│
├── 2.3 Viscoelastic Constitutive Models
│   ├── Hookean dumbbell kinetic theory & Oldroyd-B formulation
│   ├── Upper-Convected Time Derivative (UCTD) & Material Frame Indifference
│   ├── Limitations of Oldroyd-B (shear plateau, extensional singularity at Wi=0.5)
│   └── Generalized regularized models: PTT (\varepsilon) and Giesekus (\alpha)
│
├── 2.4 Nondimensionalization & Governing Equations
│   ├── Reference scales (H_ref, U_ref, inertial pressure \rho U_ref^2)
│   ├── Dimensionless groups: Reynolds (Re), Weissenberg (Wi), Viscosity ratio (\beta)
│   ├── Dimensionless vector PDE system
│   └── Explicit 2D Cartesian PDE residuals (2 momentum + 3 constitutive)
│
├── 2.5 Four-Roll Mill Benchmark & Boundary Conditions
│   ├── Geometry, roller counter-rotation (\Omega_k), and hyperbolic stagnation point
│   ├── Planar extensional kinematics (u = \dot{\epsilon}x, v = -\dot{\epsilon}y)
│   ├── Roller wall no-slip BCs and polymeric stress Dirichlet anchoring
│   └── Enclosed domain pressure gauge fix (single-point pressure pin p(x_0)=0)
│
└── 2.6 (Recommended) Stream Function Inductive Bias Formulation
    ├── Stream function definition \psi(x, y) satisfying \nabla \cdot u = 0 analytically
    └── Multi-head PINN mapping: (x,y) -> (\psi, p, \tau_{xx}, \tau_{xy}, \tau_{yy})
```

---

## 3. Ready-to-Use BibTeX Entries for `references.bib`

```bibtex
@book{bird1987dynamics,
  title={Dynamics of Polymeric Liquids, Volume 1: Fluid Mechanics},
  author={Bird, R. Byron and Armstrong, Robert C. and Hassager, Ole},
  year={1987},
  publisher={John Wiley \& Sons},
  edition={2nd},
  doi={10.1002/cite.330600823},
  isbn={978-0471802457}
}

@book{deen2012transport,
  title={Analysis of Transport Phenomena},
  author={Deen, William M.},
  year={2012},
  publisher={Oxford University Press},
  edition={2nd},
  isbn={978-0199740284}
}

@book{owens2002computational,
  title={Computational Rheology},
  author={Owens, Robert G. and Phillips, Timothy N.},
  year={2002},
  publisher={Imperial College Press},
  doi={10.1142/p160}
}

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

@book{bird2007transport,
  title={Transport Phenomena},
  author={Bird, R. Byron and Stewart, Warren E. and Lightfoot, Edwin N.},
  year={2007},
  publisher={John Wiley \& Sons},
  edition={2nd / Revised},
  isbn={978-0470115398}
}

@article{taylor1934formation,
  title={The formation of emulsions in definable fields of flow},
  author={Taylor, Geoffrey I.},
  journal={Proceedings of the Royal Society of London. Series A},
  volume={146},
  number={858},
  pages={501--523},
  year={1934},
  doi={10.1098/rspa.1934.0169}
}

@article{bentley1986experimental,
  title={An experimental investigation of drop deformation and breakup in steady, two-dimensional linear flows},
  author={Bentley, B. J. and Leal, L. G.},
  journal={Journal of Fluid Mechanics},
  volume={167},
  pages={241--283},
  year={1986},
  doi={10.1017/S0022112086002811}
}
```

---

## Related Concepts
- **Topics**: [[Fluid_Dynamics]], [[Viscoelasticity]], [[Nondimensionalization]], [[Upper-convected time derivative]], [[Pressure_Stress_Decoupling]]
- **Literature**: [[Deen_Analysis_of_Transport_Phenomena]], [[Bird_Armstrong_Hassager_Dynamics_of_Polymer_Liquids]], [[Owens_Phillips_Computational_Rheology]], [[Oldroyd_1950_Rheological_Equations_of_State]], [[Bird_Stewart_Lightfoot_Transport_Phenomena]]
- **Systems**: [[Viscoelastic_Fluids]], [[Viscoelastic_Training]]
- **Methods**: [[ViscoelasticNet]], [[ViscoelasticNet_Full model]], [[Staged_Training_Procedure]], [[Pressure_Point_Anchoring]]
