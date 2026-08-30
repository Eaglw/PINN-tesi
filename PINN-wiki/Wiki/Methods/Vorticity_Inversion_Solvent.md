# Method: Vorticity Inversion for Solvent Viscosity ($\\eta_s$)

## Overview

In the staged inverse PINN framework for viscoelastic flows (e.g., the Oldroyd-B four-roll mill), identifying the solvent viscosity $\\eta_s$ (or non-dimensional $\\mu_s^*$) in Phase 2 typically relies on the linear momentum balance:
 \\mu_s^* \\nabla^2 \\mathbf{u} + \\nabla \\cdot \\boldsymbol{\\tau} - \\nabla p - Re_{\\text{scale}} (\\mathbf{u} \\cdot \\nabla)\\mathbf{u} = \\mathbf{0} 

However, simultaneous optimization of the scalar parameter $\\mu_s^*$ and the high-dimensional continuous pressure field (x,y)$ suffers from a fundamental **gauge degeneracy (mutual feedback loop)**. This page documents:
1. The mathematical origin of the monotonic upward drift of $\\eta_s$.
2. The **Vorticity Transport / Curl Formulation** as a decoupled method to identify $\\eta_s$ independently of pressure.
3. The **Staged Pressure-Warmup Protocol** as a computationally efficient proxy.

---

## 1. The Gauge Degeneracy: Why $\\eta_s$ Drifts Monotonically

### The Lack of Pressure Scale Anchoring
In the standard experimental setup, pressure is constrained only by a single Dirichlet point on the boundary:
 p(x_0, y_0) = 0 
This fixes the integration constant $+C$, but provides **zero boundary constraint on the amplitude/scale** of $\\nabla p$.

### The Mutual Feedback Loop
Setting the derivative of the momentum loss $\\mathcal{L}_{\\text{mom}} = \\|\\mu_s^* \\nabla^2 \\mathbf{u} + \\nabla\\cdot\\boldsymbol{\\tau} - \\nabla p\\|^2$ with respect to $\\mu_s^*$ to zero yields the instantaneous stationary point:
 \\mu_s^{*, \\text{opt}} = \\frac{\\int_\\Omega (\\nabla p - \\nabla\\cdot\\boldsymbol{\\tau}) \\cdot \\nabla^2 \\mathbf{u} \\, d\\Omega}{\\int_\\Omega \\|\\nabla^2 \\mathbf{u}\\|^2 \\, d\\Omega} 

When \model_p\ is initialized from zero at the beginning of Phase 2:
1. As the neural network learns, the amplitude and variance of $\\nabla p$ grow monotonically.
2. The inner product $\\int_\\Omega \\nabla p \\cdot \\nabla^2 \\mathbf{u} \\, d\\Omega$ grows proportionally with the scale of \model_p\.
3. Consequently, $\\mu_s^*$ is pulled upward.
4. An increased $\\mu_s^*$ increases the target magnitude for $\\nabla p$, which encourages \model_p\ to inflate further.

This creates a **symbiotic inflation loop**: $\\mu_s$ and $\\nabla p$ increase together while the algebraic loss $\\mathcal{L}_{\\text{mom}}$ actually decreases. 
- *Experimental verification*: At the end of unconstrained Adam Phase 2, $\\mu_s$ was overestimated by **$+38\\%$** (.138\\text{ Pa}\\cdot\\text{s}$ vs .100\\text{ Pa}\\cdot\\text{s}$), and the standard deviation of predicted pressure $\\nabla p$ was overestimated by **$+40\\%$**!
- *Conclusion*: Crossing the ground truth (.100$) during training is merely a **transient crossing**, not a true local minimum of the loss landscape.

---

## 2. The Vorticity / Curl Formulation ($\\nabla \\times$)

To break the mutual dependence between $\\mu_s^*$ and (x,y)$, one applies the curl operator ($\\nabla \\times$) to the momentum equation. Because the curl of any gradient field is identically zero:
 \\nabla \\times (\\nabla p) \\equiv \\mathbf{0} 

The pressure field **completely drops out of the equation**, yielding the steady viscoelastic vorticity transport equation:
 \\mu_s^* \\nabla^2 \\omega_z = - \\nabla \\times (\\nabla \\cdot \\boldsymbol{\\tau}) + Re_{\\text{scale}} \\nabla \\times ((\\mathbf{u} \\cdot \\nabla)\\mathbf{u}) 
where $\\omega_z = \\frac{\\partial v}{\\partial x} - \\frac{\\partial u}{\\partial y} = -\\nabla^2 \\psi$ is the scalar 2D vorticity.

### Theoretical Advantages for Inverse Problems
1. **Zero Pressure Coupling**: $\\mu_s^*$ is the **sole unknown** in the equation.
2. **Strictly Convex Quadratic Landscape**: The loss $\\mathcal{L}_{\\text{curl}} = \\|\\mu_s^* \\nabla^2 \\omega_z + \\nabla \\times (\\nabla \\cdot \\boldsymbol{\\tau}) - Re \\dots\\|^2$ is a 1D convex parabola with a single global analytical minimum:
    \\mu_s^{*, \\text{exact}} = \\frac{\\int_\\Omega \\left[ -\\nabla \\times (\\nabla \\cdot \\boldsymbol{\\tau}) + Re \\nabla \\times ((\\mathbf{u}\\cdot\\nabla)\\mathbf{u}) \\right] \\cdot \\nabla^2 \\omega_z \\, d\\Omega}{\\int_\\Omega (\\nabla^2 \\omega_z)^2 \\, d\\Omega} 
3. **Noise Immunity via PINN Autograd**: While finite difference evaluations of $\\nabla^2 \\omega_z$ (which require 4th-order spatial derivatives) amplify discrete mesh noise, neural network automatic differentiation (Autograd) operates on the smooth analytical manifold learned in Phase 1.

### Future Development Pathway
While evaluating biharmonic operators ($\\nabla^4 \\psi$) is computationally demanding on large collocation sets, it can be evaluated:
- Post-hoc as a single-step algebraic projection after Phase 1.
- Or as an auxiliary scalar calibration loss during Phase 2 warmup.

---

## 3. Practical Proxy: The Staged Phase 2 Warmup Protocol

Because evaluating 4th-order autograd derivatives during continuous training carries non-trivial GPU memory overhead, an effective and practical proxy is the **Staged Pressure-Warmup Protocol**:

1. **Phase 2A (Pressure Field Formation / Warmup)**:
   - $\\mu_s$ is **frozen** at its initial guess (e.g., .080\\text{ Pa}\\cdot\\text{s}$).
   - \model_p\ is optimized with Adam until its spatial lobes and gradient morphology reach dynamic equilibrium with the divergence of stress $\\nabla\\cdot\\boldsymbol{\\tau}$.
2. **Phase 2B (Solvent Identification)**:
   - $\\mu_s$ is **unlocked** in Adam. Because $\\nabla p$ already has a stabilized scale and non-zero morphology, $\\mu_s$ is optimized towards true momentum balance without triggering the initial inflation cascade.
3. **Phase 2C (High-Precision Physical Refinement)**:
   - $\\mu_s$ is **frozen** at the value identified in Phase 2B.
   - L-BFGS in FP64 refines the pressure field (x,y)$ and cleans up the residual without parameter drift.

---

## References & Back-links
- [[Viscoelastic_Parameter_Identifiability]] (Full-blind parameter identification in Oldroyd-B)
- [[Vorticity_Regularization]] (Vorticity transport as forward regularization in Phase 1)
- [[Pressure_Stress_Decoupling]] (Helmholtz-Hodge decomposition and pressure isolation)
- [[Staged_Training_Procedure]] (Multi-stage training architecture)
- [[Viscoelastic_Training]] (Four-roll mill experiment guide)
