# Topic: Pressure-Stress Decoupling

## Overview
In the fluid dynamics of complex and viscoelastic fluids, the total Cauchy stress tensor $\boldsymbol{\sigma}$ at any point is decomposed into isotropic and deviatoric components:

$$ \boldsymbol{\sigma} = -p \mathbf{I} + \mathbf{T} $$

1. **Isotropic (Spherical) Part**: $-p \mathbf{I}$, where $p$ is the hydrostatic (or thermodynamic) pressure and $\mathbf{I}$ is the identity tensor. This component acts uniformly in all directions and only affects volume change (compression or expansion) without changing the shape.
2. **Deviatoric (Extra-Stress) Part**: $\mathbf{T}$, representing shear stresses and normal stress differences. It is responsible for the deformation of shape (shear/flow).

For complex fluids, the extra-stress $\mathbf{T}$ is split into a Newtonian solvent contribution and a polymeric elastic contribution $\boldsymbol{\tau}$:

$$ \mathbf{T} = 2\eta_s \mathbf{D} + \boldsymbol{\tau} $$

where $\mathbf{D} = \frac{1}{2} \left( \nabla \mathbf{u} + (\nabla \mathbf{u})^T \right)$ is the rate-of-strain tensor and $\eta_s$ is the solvent viscosity.

### Physical and Mathematical Decoupling
The decoupling of the pressure $p$ and the polymeric stress $\boldsymbol{\tau}$ arises from the **incompressibility assumption** ($\nabla \cdot \mathbf{u} = 0$).

In an incompressible fluid:
- The pressure $p$ loses its thermodynamic connection to density (there is no equation of state linking them) and acts strictly as a **Lagrange multiplier** to enforce the divergence-free velocity constraint.
- The polymeric stress $\boldsymbol{\tau}$ depends entirely on the deformation history of the fluid. The constitutive equations (such as Oldroyd-B, Giesekus, or Phan-Thien-Tanner) govern the evolution of $\boldsymbol{\tau}$ as a function of the velocity gradient $\nabla \mathbf{u}$. **The pressure $p$ never appears in these constitutive equations.**

---

## Technical Implementation & Physical Details
In Physics-Informed Neural Networks (PINNs) such as [[ViscoelasticNet]], this physical decoupling is directly translated into the neural architecture and loss formulation to structure and stabilize the learning process:

* **Separate Outputs**: The neural network outputs separate fields for pressure $p$ and the independent components of the polymeric stress tensor $\boldsymbol{\tau}$ (e.g., $\tau_{xx}, \tau_{xy}, \tau_{yy}$).
* **Constitutive Loss**: The rheological/constitutive loss (e.g., Giesekus or Oldroyd-B residuals) penalizes the network based strictly on the relationship between $\boldsymbol{\tau}$ and velocity $\mathbf{u}$. This part of the optimization is entirely independent of $p$.
* **Momentum Conservation Loss**: The momentum conservation equation (generalized Navier-Stokes) is the only place where $p$ and $\boldsymbol{\tau}$ couple, as the gradient of the isotropic pressure must balance the divergence of the extra-stress and the inertial/viscous forces:

$$ \rho \frac{D\mathbf{u}}{Dt} = -\nabla p + \eta_s \nabla^2 \mathbf{u} + \nabla \cdot \boldsymbol{\tau} $$

This separation isolates the viscoelastic rheological response of the fluid from the isotropic pressure force balances, improving convergence and parameter identification.

## References & Back-links
- [[Viscoelasticity]]
- [[ViscoelasticNet]]
- [[Viscoelastic_Fluids]]
- [[Thakur_et_al_ViscoelasticNet]]
