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


## The Helmholtz-Hodge Pressure Inference Limit

In an incompressible flow, pressure $p$ is governed by the momentum equation:
$$ \nabla p = - Re (\mathbf{u} \cdot \nabla \mathbf{u}) + \beta \nabla^2 \mathbf{u} + \nabla \cdot \boldsymbol{\tau} $$

Let the right-hand side be a vector field $\mathbf{f}(\mathbf{u}, \boldsymbol{\tau})$:
$$ \nabla p = \mathbf{f} $$

For a scalar field $p$ to exist such that the momentum residual is exactly zero, the vector field $\mathbf{f}$ must be conservative (irrotational). Mathematically, this requires:
$$ \nabla \times \mathbf{f} = 0 \quad \iff \quad \frac{\partial f_y}{\partial x} - \frac{\partial f_x}{\partial y} = 0 $$

### Helmholtz-Hodge Decomposition
According to the Helmholtz-Hodge theorem, any vector field $\mathbf{f}$ on a bounded domain can be uniquely decomposed into a curl-free (conservative) component and a divergence-free (solenoidal/rotational) component:
$$ \mathbf{f} = \nabla p_{\text{true}} + \mathbf{g}, \quad \text{with } \nabla \cdot \mathbf{g} = 0 $$

The momentum loss minimized by the PINN is:
$$ L_{\text{momentum}} = \frac{1}{2} \int_\Omega \|\nabla p - \mathbf{f}\|^2 d\Omega = \frac{1}{2} \int_\Omega \|\nabla(p - p_{\text{true}})\|^2 d\Omega + \frac{1}{2} \int_\Omega \|\mathbf{g}\|^2 d\Omega $$

Since the pressure network `model_p` can only represent a gradient field, it can at best fit the conservative part $p_{\text{true}}$, reducing the first term to zero. The second term, representing the rotational component $\|\mathbf{g}\|^2$, remains as a **constant residual bottleneck** as long as velocity $\mathbf{u}$ and stress $\boldsymbol{\tau}$ are frozen.

### Noise Amplification in Frozen Training
When $\mathbf{u}$ (or stream function $\psi$) and $\boldsymbol{\tau}$ are predicted by frozen, pre-trained neural networks, they inevitably contain small approximation errors (typically $1\% - 5\%$ L2 error). 
Because the term $\mathbf{f}$ contains high-order spatial derivatives (up to second-order derivatives of velocity $\mathbf{u}$, which translate to **third-order derivatives** of $\psi$):
* Numerical differentiation acts as a high-pass filter.
* Even tiny, smooth errors in $\psi$ are amplified dramatically in its third derivatives (often exceeding $50\% - 100\%$ local error).
* This amplifies the curl of the error, leading to a large non-zero rotational component $\mathbf{g}$.
* Since `model_p` cannot fit this rotational noise, the loss becomes stuck at a high value.

### Resolution via Joint Velocity-Pressure Training
Unfreezing `model_psi` (velocity) during Phase 2 allows the optimizer to backpropagate momentum residuals to the stream function. By making minute, high-frequency adjustments to $\psi$ (often $<0.1\%$ change in L2 velocity error), the optimizer eliminates the rotational component of the noise, forcing $\mathbf{g} \to 0$. This aligns the velocity fields with the pressure gradient, allowing the momentum loss to drop and the pressure field to converge.

### Alternative Resolution: Vorticity Regularization in Phase 1
Instead of unfreezing $\psi$ in Phase 2, one can prevent the rotational noise from forming in the first place by adding the [[Vorticity_Regularization|vorticity transport equation]] as a regularizer during Phase 1. Since $\nabla \times \nabla p \equiv 0$, the vorticity equation constrains the higher-order derivatives of $\psi$ without involving pressure, ensuring $\mathbf{F}$ is nearly conservative when Phase 2 begins. See [[Vorticity_Regularization]] for full details.

## References & Back-links
- [[Viscoelasticity]]
- [[ViscoelasticNet]]
- [[Viscoelastic_Fluids]]
- [[Thakur_et_al_ViscoelasticNet]]
- [[Staged_Training_Procedure]]
- [[Viscoelastic_Training]]
- [[Vorticity_Regularization]]
