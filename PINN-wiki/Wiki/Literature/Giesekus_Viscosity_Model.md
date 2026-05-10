---
title: "Viscosity model based on Giesekus equation"
source: "[Viscosity model based on Giesekus equation.md](file:///c:/Users/eaglw/Documents/PINN%20tesi/PINN-wiki/Reference/Viscosity%20model%20based%20on%20Giesekus%20equation.md)"
author: "Sun Kyoung Kim"
published: 2024-01-01
type: "paper"
---

## Summary
Introduces a new viscosity model derived from the Giesekus differential constitutive equation. The model is shown to be more flexible than Cross or Carreau models, particularly in capturing the inflection point of the shear-thinning curve for viscoelastic fluids like polystyrene.

## Key Methodology
- **Derived Model**: Based on the Giesekus mobility parameter $\alpha$.
- **Parameters**: 
    - $\eta_0$ (zero-shear viscosity)
    - $\eta_\infty$ (infinite-shear viscosity)
    - $n$ (power-law index)
    - $\lambda$ (characteristic time)
    - $\alpha$ (mobility parameter - unique to this model)
    - $a$ (Yasuda-type curvature parameter)
- **Validation**: Fitted against linear polystyrene and MWCNT-filled polypropylene data.

## Key Findings
- **Mobility Parameter ($\alpha$)**: Distinctive role in adjusting the inflection shape of the viscosity curve. Values above 0.5 allow for steeper shear-thinning than the power-law model.
- **Versatility**: The model matches experimental data where Carreau overestimates and Cross underestimates at the inflection point.
- **Applicability**: Shown to work for both polymeric liquids and particulate slurries (e.g., anode slurries).

## Related
- [[Viscoelasticity]]
- [[Fluid_Dynamics]]
- [[Thakur_et_al_ViscoelasticNet]]
