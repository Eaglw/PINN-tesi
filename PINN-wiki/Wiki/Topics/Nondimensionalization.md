# Topic: Nondimensionalization

## Overview
Nondimensionalization is the mathematical process of removing physical units from governing equations by scaling independent and dependent variables with characteristic physical constants of the system. 

In Physics-Informed Neural Networks (PINNs), training directly on variables with physical units (which can span several orders of magnitude, e.g., coordinate fields in millimeters vs. pressures in Pascals) introduces severe numerical challenges:
- **Ill-conditioned Hessian matrices**, which degrade optimization performance.
- **Gradient imbalance** across different loss components (e.g., boundary conditions vs. physical residuals).
- **Slow convergence** or entrapment in poor local minima.

Scaling all input coordinates and physical variables to dimensionless ranges of $O(1)$ (typically within $[0, 1]$ or $[-1, 1]$) ensures a balanced, well-behaved loss landscape and enhances training stability.

---

## Technical Implementation & Physical Details

The Viscoelastic PINN framework (`Viscoelastic/`) implements a rigorous, two-fold nondimensionalization strategy:

### 1. Navier-Stokes (Momentum Conservation)
The dimensional, steady-state Cauchy momentum equation for a viscoelastic fluid (neglecting gravity) is:
$$ \rho (\mathbf{v}^* \cdot \nabla^* \mathbf{v}^*) = -\nabla^* p^* + \mu_s \nabla^{*2} \mathbf{v}^* + \nabla^* \cdot \boldsymbol{\tau}_p^* $$
where the asterisks ($*$) denote dimensional quantities.

To nondimensionalize this system under confined flow conditions (e.g., microfluidic channels, polymer extrusion), the codebase employs **Viscous Scaling** for the pressure. Instead of scaling the pressure on kinetic energy ($\rho U^2$), which is standard in high-Reynolds aerodynamics, it is scaled on the total viscous stresses:
$$ p = \frac{p^* H}{\mu_{tot} U} $$
where:
* $H$ is the characteristic channel half-width.
* $U$ is the characteristic velocity scale.
* $\mu_{tot} = \mu_s + \mu_p$ is the total viscosity (solvent viscosity $\mu_s$ plus polymeric viscosity $\mu_p$).

Dividing the Cauchy equation by the viscous stress scale $\frac{\mu_{tot} U}{H^2}$ yields:
$$ \frac{\rho U H}{\mu_{tot}} (\mathbf{v} \cdot \nabla \mathbf{v}) = -\nabla p + \frac{\mu_s}{\mu_{tot}} \nabla^2 \mathbf{v} + \nabla \cdot \left( \frac{\boldsymbol{\tau}_p^* H}{\mu_{tot} U} \right) $$

By defining the classical dimensionless parameters:
* **Reynolds Number**: $Re = \frac{\rho U H}{\mu_{tot}}$
* **Viscosity Ratio**: $\beta = \frac{\mu_s}{\mu_{tot}}$

The momentum equation reduces to:
$$ Re (\mathbf{v} \cdot \nabla \mathbf{v}) = -\nabla p + \beta \nabla^2 \mathbf{v} + \nabla \cdot \boldsymbol{\tau}_p $$
where $\boldsymbol{\tau}_p = \frac{\boldsymbol{\tau}_p^* H}{\mu_{tot} U}$.

#### Polymeric Stress Rescaling
To prevent numerical instability in flows where $\beta \to 1$ (highly solvent-dominated regimes), the neural network is parameterized to predict a **rescaled polymeric stress tensor** $\tilde{\boldsymbol{\tau}}$ instead of the raw $\boldsymbol{\tau}_p$:
$$ \boldsymbol{\tau}_p^* = \frac{\mu_p U}{H} \tilde{\boldsymbol{\tau}} $$
Since $\frac{\mu_p}{\mu_{tot}} = 1 - \beta$, the relationship between the two dimensionless stresses is:
$$ \boldsymbol{\tau}_p = (1 - \beta) \tilde{\boldsymbol{\tau}} $$

Substituting this back into the momentum balance yields the exact formulation implemented in `Viscoelastic_physics.py`:
$$ Re (\mathbf{v} \cdot \nabla \mathbf{v}) = -\nabla p + \beta \nabla^2 \mathbf{v} + (1 - \beta) \nabla \cdot \tilde{\boldsymbol{\tau}} $$

In code, this maps to the momentum residual `f_u` (and analogously `f_v`):
```python
f_u = Re * (u * u_x + v * u_y) + p_x - beta * (u_xx + u_yy) - one_m_beta * (tt_xx_x + tt_xy_y)
```

### 2. Constitutive Equations (PTT-Giesekus)
The dimensional non-linear constitutive equation coupling Phan-Thien-Tanner (PTT) and Giesekus models is:
$$ f_{PTT}(\boldsymbol{\tau}_p^*) \boldsymbol{\tau}_p^* + \lambda \overset{\nabla}{\boldsymbol{\tau}_p^*} + \alpha \frac{\lambda}{\mu_p} \boldsymbol{\tau}_p^{*2} = 2 \mu_p \mathbf{D}^* $$
where:
* $\lambda$ is the relaxation time.
* $\alpha$ is the Giesekus mobility parameter.
* $\mathbf{D}^* = \frac{1}{2}(\nabla^* \mathbf{v}^* + (\nabla^* \mathbf{v}^*)^T)$ is the deformation rate tensor.
* $f_{PTT}(\boldsymbol{\tau}_p^*) = 1 + \epsilon \frac{\lambda}{\mu_p} \text{tr}(\boldsymbol{\tau}_p^*)$ is the linear PTT function.

Substituting the rescaled stress relation $\boldsymbol{\tau}_p^* = \frac{\mu_p U}{H} \tilde{\boldsymbol{\tau}}$ and dividing the entire equation by the scale factor $\frac{\mu_p U}{H}$ yields:
$$ f_{PTT}(\tilde{\boldsymbol{\tau}}) \tilde{\boldsymbol{\tau}} + Wi \overset{\nabla}{\tilde{\boldsymbol{\tau}}} + \alpha Wi \tilde{\boldsymbol{\tau}}^2 = 2 \mathbf{D} $$
where $Wi = \frac{\lambda U}{H}$ is the **Weissenberg Number** (represented in code calculations by the dimensionless parameter `Wi`), and $\mathbf{D}$ is the dimensionless strain rate tensor.

This mathematical transformation maps term-by-term in the code:
* **PTT Factor ($f_{PTT}$)**: 
  $$ f_{PTT} = 1 + \epsilon Wi \cdot \text{tr}(\tilde{\boldsymbol{\tau}}) $$
  Code: `f_PTT = 1.0 + eps * Wi * (tt_xx + tt_yy)`.
* **Upper-Convected Derivative ($\overset{\nabla}{\tilde{\boldsymbol{\tau}}}$)**:
  For the $xx$ component:
  $$ \overset{\nabla}{\tilde{\tau}}_{xx} = u \frac{\partial \tilde{\tau}_{xx}}{\partial x} + v \frac{\partial \tilde{\tau}_{xx}}{\partial y} - 2 \frac{\partial u}{\partial x} \tilde{\tau}_{xx} - 2 \frac{\partial u}{\partial y} \tilde{\tau}_{xy} $$
  Code: `upper_xx = (u * tt_xx_x + v * tt_xx_y - 2 * u_x * tt_xx - 2 * u_y * tt_xy)`.
* **Giesekus Term ($\tilde{\boldsymbol{\tau}}^2$)**:
  For a symmetric tensor in 2D, the tensor multiplication $(\tilde{\boldsymbol{\tau}}^2)_{xy}$ is:
  $$ (\tilde{\boldsymbol{\tau}}^2)_{xy} = \tilde{\tau}_{xx}\tilde{\tau}_{xy} + \tilde{\tau}_{xy}\tilde{\tau}_{yy} = \tilde{\tau}_{xy}(\tilde{\tau}_{xx} + \tilde{\tau}_{yy}) $$
  Code: `alpha * Wi * tt_xy * (tt_xx + tt_yy)`.
* **Deformation Rate Tensor ($2\mathbf{D}$)**:
  For the $xy$ component:
  $$ 2D_{xy} = \frac{\partial u}{\partial y} + \frac{\partial v}{\partial x} $$
  Code: `- (u_y + v_x)`.

---

## References & Back-links
- [[Hazra_et_al_Convective_Heat_Transfer]] (Initial scaling discussion)
- [[ViscoelasticNet_Full model]] (Implementation of unified rheology)
- [[Viscoelastic_Fluids]] (Physical systems and domain dynamics)
- [[Viscoelastic_Residual_Scaling]] (Detailed PDE residual scaling and balancing analysis)
