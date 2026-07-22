# Topic: Viscoelastic Parameter Identifiability

## Overview
Analysis of parameter sensitivity and ill-conditioned inversion in viscoelastic PINNs (specifically Oldroyd-B models). It highlights the non-linear transformation between the physical polymer viscosity $\eta_p$ (`etap`), the relaxation time $\lambda$ (`lam`), and the non-dimensional parameter $\beta_{poly}$.

## Mathematical Analysis: $\lambda$ vs. $\beta_{poly}$ ($\eta_p$)
In the non-dimensional Oldroyd-B constitutive equation:
$$\mathbf{\tau} + \lambda \mathbf{\tau}_{(1)} = 2 \beta_{poly} \mathbf{D}$$
where $\beta_{poly} = \frac{\eta_p}{\eta_s + \eta_p}$.

### 1. Direct Sensitivity of Relaxation Time $\lambda$
The relaxation time $\lambda$ multiplies the upper-convective rate tensor $\mathbf{\tau}_{(1)}$ linearly through the Weissenberg number $Wi = \lambda \frac{U_{ref}}{H_{ref}}$:
$$\frac{\partial Wi}{\partial \lambda} = \frac{U_{ref}}{H_{ref}} \approx 1.6666$$
Because the spatial pattern of $\mathbf{\tau}_{(1)}$ is tightly constrained by kinematics and velocity streamlines, $\lambda$ experiences strong, direct gradient updates and converges rapidly (e.g., L2 error $< 3.8\%$).

### 2. Ill-Conditioned Sensitivity of Polymeric Viscosity $\eta_p$
Unlike $\lambda$, the polymeric viscosity $\eta_p$ enters non-linearly through the bounded ratio $\beta_{poly}(\eta_p)$:
$$\beta_{poly}(\eta_p) = \frac{\eta_p}{\eta_s + \eta_p}$$

Evaluating the derivative at ground truth ($\eta_s = 0.1, \eta_p = 0.9$):
$$\frac{d \beta_{poly}}{d \eta_p} = \frac{\eta_s}{(\eta_s + \eta_p)^2} = \frac{0.1}{(1.0)^2} = 0.1$$

For $\eta_p$ growing from $0.900$ to $1.490$:
- Ground truth: $\beta_{poly, true} = \frac{0.9}{0.1 + 0.9} = 0.9000$
- Learned value: $\beta_{poly, learned} = \frac{1.49}{0.1 + 1.49} = 0.9371$
- Difference in PDE residual space: $\Delta \beta_{poly} = 0.0371 \quad (\mathbf{3.7\%} \text{ error!})$

> [!IMPORTANT]
> **Key Insight**: A tiny **3.7% relative error** in the predicted stress field $\mathbf{\tau}$ (or $\beta_{poly}$) produces a **65.6% error** in $\eta_p$ due to the flattening slope of $\beta_{poly}(\eta_p)$ near 1.0. The PDE equation "sees" the fluid through $\beta_{poly}$, so a near-perfect stress field match ($\sim 3\%$ L2 error) corresponds to a large scalar drift on $\eta_p$.

### 3. Gradient Flattening & Countermeasures
As $\eta_p$ grows, $\frac{d \beta_{poly}}{d \eta_p} = \frac{0.1}{(0.1 + \eta_p)^2}$ drops to $0.039$, causing severe gradient vanishing for $\eta_p$ under Adam with small learning rates ($LR_{param} = 2 \times 10^{-5}$).

**Countermeasures**:
1. **Adaptive Learning Rates**: Increase `PARAM_LR_FACTOR` (or use dedicated $LR_{param} \approx 10^{-3}$) to compensate for the $0.039$ gradient dampening factor.
2. **Momentum Equation Activation (Phase 2)**: Re-enabling Navier-Stokes momentum ($\nabla p = \eta_s \nabla^2 \mathbf{u} + \nabla \cdot \mathbf{\tau}$) anchors the total viscosity $\eta_{tot} = \eta_s + \eta_p$ via pressure drop $\nabla p$, eliminating the plateau.
3. **Curvature Optimization (L-BFGS Phase 1.5/3)**: L-BFGS uses second-order curvature estimates (inverse Hessian approximation), automatically scaling step sizes along flat parameter directions.

## Related Concepts
- **Topics**: [[Inverse_Problems]], [[Nondimensionalization]], [[Viscoelasticity]], [[Pressure_Stress_Decoupling]]
- **Methods**: [[ViscoelasticNet]], [[Staged_Training_Procedure]], [[Staged_Precision_Strategy]]
