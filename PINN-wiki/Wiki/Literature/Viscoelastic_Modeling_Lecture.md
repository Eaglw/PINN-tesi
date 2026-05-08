---
title: "Viscoelastic modeling"
source: "[[Viscoelastic modeling.md]]"
author: "Lecture Notes"
type: "notes"
---

## Summary
Comprehensive lecture notes on the modeling and simulation of non-Newtonian fluids, covering Newtonian foundations, typical non-Newtonian phenomena, and various constitutive models for generalized Newtonian, viscoplastic, and viscoelastic fluids.

## Key Methodology
- **Classification of Fluids**:
    - **Newtonian**: Constant viscosity, no memory.
    - **Generalized Newtonian**: Shear-dependent viscosity (Power-law, Carreau).
    - **Viscoplastic**: Yield stress (Bingham, Papanastasiou).
    - **Viscoelastic**: Elasticity + Viscosity, memory effects, normal stresses (UCM, Oldroyd-B, Giesekus, PTT).
- **Dimensionless Numbers**:
    - **Reynolds (Re)**: Inertia vs Viscosity.
    - **Bingham (Bn)**: Yield stress vs Viscous stress.
    - **Weissenberg (Wi)**: Elastic time vs Flow time ($\lambda U / D$).
    - **Viscosity Ratio ($\eta_r$)**: Solvent vs Total viscosity.

## Key Findings
- **Viscoelastic Stability**: Numerical instabilities often occur for $Wi > 1$ (High Weissenberg Number Problem).
- **Decoupling**: Decoupled schemes (e.g., D'Avino and Hulsen) can reduce computational effort.
- **Normal Stresses**: $N_1$ (positive) and $N_2$ (negative, smaller) are key indicators of viscoelasticity.
- **Trouton Ratio**: $\eta_{el} / \eta = 3$ for Newtonian fluids; varies for non-Newtonian.

## Related
- [[Viscoelasticity]]
- [[Fluid_Dynamics]]
- [[Nondimensionalization]]
- [[Viscoelastic_Fluids]]
- [[ViscoelasticNet]]
