# Topic: Viscoelasticity

## Overview
Viscoelasticity is the study of materials that exhibit both viscous and elastic characteristics when undergoing deformation. Unlike purely elastic materials, which store energy and return to their original shape instantly, viscoelastic materials dissipate energy (hysteresis) and show time-dependent behavior such as [[Creep]] and [[Stress_Relaxation]].

Core concepts include:
- **Elasticity**: Modeled as springs (Hooke's Law: $\sigma = E \epsilon$).
- **Viscosity**: Modeled as dashpots (Newton's Law: $\sigma = \eta \dot{\epsilon}$).
- **Deborah Number**: $De = \lambda / t$, where $\lambda$ is the material relaxation time.

## Technical Implementation
In the context of PINNs, viscoelasticity is modeled by incorporating constitutive equations into the loss function. Common models include:
- **Linear**: Maxwell, Kelvin-Voigt, and Zener models.
- **Nonlinear**: 
    - **UCM (Upper-Convected Maxwell)**: Simplest observer-invariant model, assumes constant viscosity.
    - **Oldroyd-B**: Adds a Newtonian solvent contribution to UCM.
    - **Giesekus**: Includes a mobility parameter $\alpha$ to capture shear-thinning and realistic inflection points.
    - **PTT (Phan-Thien Tanner)**: Uses a trace-dependent function to avoid infinite stress in elongational flows.

The challenge in PINN implementations (e.g., [[ViscoelasticNet]]) is the coupling of momentum conservation with these constitutive equations, often requiring multi-network architectures to handle velocity, pressure, and stress components separately.

## References
- [[Viscoelasticity_Theory]]
- [[Viscoelastic_Fluids]]
- [[Upper_Convected_Maxwell]]
- [[Giesekus_Viscosity_Model]]
- [[Viscoelastic_Modeling_Lecture]]
- [[Note_05_Academic_Context]]
