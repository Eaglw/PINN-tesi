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

The Viscoelastic PINN framework implements a rigorous nondimensionalization strategy across momentum, constitutive, and geometric domains.

### 1. Navier-Stokes (Momentum Conservation) & Scale Decoupling
The dimensional, steady-state Cauchy momentum equation for an incompressible viscoelastic fluid (neglecting gravity) is:
$$ \rho (\mathbf{v}^* \cdot \nabla^* \mathbf{v}^*) = -\nabla^* p^* + \eta_s \nabla^{*2} \mathbf{v}^* + \nabla^* \cdot \boldsymbol{\tau}_p^* $$
where the asterisks ($*$) denote dimensional quantities.

#### Forward vs. Inverse Scaling Strategies
- **Forward Problems (Known Parameters)**: Pressure is scaled on the total viscous scale $\frac{\eta_{\text{tot}} U}{H}$, leading to the standard dimensionless momentum equation:
  $$ Re (\mathbf{v} \cdot \nabla \mathbf{v}) = -\nabla p + \beta \nabla^2 \mathbf{v} + \nabla \cdot \boldsymbol{\tau}_p $$
  where $Re = \frac{\rho U H}{\eta_{\text{tot}}}$ and $\beta = \frac{\eta_s}{\eta_{\text{tot}}}$.

- **Inverse Problems (Unknown Parameters - Decoupled Scaling)**:
  When identifying unknown viscosities $(\eta_s, \eta_p)$, defining $Re$ with a trainable $\eta_{\text{tot}}$ creates an artificial degree of freedom ($\eta_{\text{tot}} \downarrow \implies Re \uparrow$), causing optimization degeneracy (observed in Run 010).
  To eliminate this, the inverse solver defines an **arbitrary reference viscosity scale** $\eta_0$:
  $$ \tilde{\eta}_s = \frac{\eta_s}{\eta_0}, \qquad \tilde{\eta}_p = \frac{\eta_p}{\eta_0}, \qquad Re_{\text{scale}} = \frac{\rho U H}{\eta_0} $$
  The resulting dimensionless momentum equation is:
  $$ Re_{\text{scale}} (\mathbf{u} \cdot \nabla \mathbf{u}) + \nabla p = \tilde{\eta}_s \nabla^2 \mathbf{u} + \nabla \cdot \boldsymbol{\tau} $$
  Here, $Re_{\text{scale}}$ is a constant numerical scale factor (not a trainable variable). $\eta_0$ is updated periodically via [[Adaptive_Nondimensionalization]]. The true physical Reynolds number $Re_{\text{phys}} = \frac{\rho U H}{\eta_s + \eta_p}$ is evaluated strictly a posteriori.

---

### 2. Constitutive Equations (Oldroyd-B / PTT / Giesekus)
The dimensional non-linear constitutive equation coupling Oldroyd-B, Phan-Thien-Tanner (PTT), and Giesekus models is:
$$ f_{PTT}(\boldsymbol{\tau}_p^*) \boldsymbol{\tau}_p^* + \lambda \overset{\nabla}{\boldsymbol{\tau}_p^*} + \alpha \frac{\lambda}{\eta_p} \boldsymbol{\tau}_p^{*2} = 2 \eta_p \mathbf{D}^* $$
where:
* $\lambda$ is the relaxation time.
* $\alpha$ is the Giesekus mobility parameter.
* $\mathbf{D}^* = \frac{1}{2}(\nabla^* \mathbf{v}^* + (\nabla^* \mathbf{v}^*)^T)$ is the deformation rate tensor.
* $f_{PTT}(\boldsymbol{\tau}_p^*) = 1 + \epsilon \frac{\lambda}{\eta_p} \text{tr}(\boldsymbol{\tau}_p^*)$ is the linear PTT function.

Scaling by the characteristic stress scale $\tau_0 = \frac{\eta_0 U}{H}$ and using $\tilde{\eta}_p = \frac{\eta_p}{\eta_0}$ yields the dimensionless form:
$$ f_{PTT}(\boldsymbol{\tau}) \boldsymbol{\tau} + Wi \overset{\nabla}{\boldsymbol{\tau}} + \alpha \frac{Wi}{\tilde{\eta}_p} \boldsymbol{\tau}^2 = 2 \tilde{\eta}_p \mathbf{D} $$
where $Wi = \lambda \frac{U}{H}$ is the **Weissenberg Number**.

---

### 3. Physical Scaling vs. Coordinate Normalization (Double Length-Scale Adimensionalization)

In confined geometries (like the 4-roll mill), the domain size $H_{\text{domain}} = 0.05 \text{ m}$ is often much larger than the local characteristic feature size, such as the roll radius $H_{\text{ref}} = R = 0.005 \text{ m}$.
To keep neural network inputs bounded in $[0, 1]^2$ while evaluating dimensionless numbers on local feature scales ($H_{\text{ref}} = 0.005 \text{ m}$):

1. **Coordinate Normalization**: Coordinates are scaled in the dataset by the domain height:
   $$ x_{\text{net}} = \frac{x_{\text{raw}} - x_{\text{min}}}{H_{\text{coord}}} \quad \in [0, 1] $$
   where $H_{\text{coord}} = 0.05 \text{ m}$.

2. **Physical Nondimensionalization**: Equations are scaled by the physical characteristic length scale:
   $$ x_{\text{phys}} = \frac{x_{\text{raw}} - x_{\text{min}}}{H_{\text{ref}}} \quad \in [0, 10] $$
   where $H_{\text{ref}} = 0.005 \text{ m}$.

3. **Stream Function Scaling**:
   $$ \psi_{\text{phys}} = \psi_{\text{net}} \cdot \left(\frac{H_{\text{coord}}}{H_{\text{ref}}}\right) $$

4. **Gradient Scaling**:
   $$ \nabla_{\text{phys}} = \nabla_{\text{net}} \cdot \left(\frac{H_{\text{ref}}}{H_{\text{coord}}}\right) = 0.1 \cdot \nabla_{\text{net}} $$

---

## References & Back-links
- [[Adaptive_Nondimensionalization]] (Block-wise adaptive scaling protocol)
- [[Viscoelastic_Parameter_Identifiability]] (Parameter sensitivity and Run 010 autopsy)
- [[ViscoelasticNet_Full model]] (Unified rheological implementation)
- [[Viscoelastic_Fluids]] (Physical systems and domain dynamics)
- [[Viscoelastic_Residual_Scaling]] (Detailed PDE residual scaling and balancing analysis)
