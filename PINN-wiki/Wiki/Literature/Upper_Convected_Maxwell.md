---
title: "Upper-convected Maxwell model"
source: "[Upper-convected Maxwell model.md](file:///c:/Users/eaglw/Documents/PINN%20tesi/PINN-wiki/Reference/Upper-convected%20Maxwell%20model.md)"
author: "Wikipedia"
type: "clipping"
---

## Summary
The **Upper-Convected Maxwell (UCM)** model is the simplest observer-independent constitutive equation for viscoelasticity. It generalizes the Maxwell material for large deformations using the **upper-convected time derivative**.

## Key Methodology
- **Constitutive Equation**: $\mathbf{T} + \lambda \stackrel{\nabla}{\mathbf{T}} = 2\eta_0 \mathbf{D}$
- **Upper-Convected Derivative**: $\stackrel{\nabla}{\mathbf{T}} = \frac{\partial}{\partial t}\mathbf{T} + \mathbf{v} \cdot \nabla \mathbf{T} - (\nabla \mathbf{v})^T \cdot \mathbf{T} - \mathbf{T} \cdot (\nabla \mathbf{v})$
- **Derivation**: Can be derived from observer invariance or mesoscopic models (Hookean Dumbbells, Temporary Networks).

## Key Findings
- **Steady Shear**: Predicts constant viscosity (linear shear stress) and non-zero first normal stress difference ($N_1 \propto \dot{\gamma}^2$), but zero second normal stress difference ($N_2 = 0$).
- **Limitations**: Predicts constant viscosity, which is unrealistic for many polymer melts (which are typically shear-thinning).
- **Elongational Flow**: Predicts elongational thickening, with viscosity approaching infinity at a critical elongational rate ($\dot{\epsilon} = 1/2\lambda$).

## Related
- [[Viscoelasticity]]
- [[Fluid_Dynamics]]
- [[Viscoelastic_Fluids]]
- [[Oldroyd_B_Model]]
