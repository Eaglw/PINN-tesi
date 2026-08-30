# Method: Vorticity Inversion for Solvent Viscosity ($\eta_s$)

## Overview

In the staged inverse PINN framework for viscoelastic flows (e.g., the Oldroyd-B four-roll mill), identifying the solvent viscosity $\eta_s$ (or non-dimensional $\mu_s^*$) in Phase 2 typically relies on the linear momentum balance:
$$\mu_s^* \nabla^2 \mathbf{u} + \nabla \cdot \boldsymbol{\tau} - \nabla p - Re_{\text{scale}} (\mathbf{u} \cdot \nabla)\mathbf{u} = \mathbf{0}$$

However, simultaneous optimization of the scalar parameter $\mu_s^*$ and the high-dimensional continuous pressure field $p(x,y)$ suffers from a fundamental **gauge degeneracy (mutual feedback loop)**. This page documents:
1. The mathematical origin of the monotonic upward drift of $\eta_s$.
2. The **Vorticity Transport / Curl Formulation** as a decoupled method to identify $\eta_s$ independently of pressure.
3. **Empirical Validation in Phase 2**: Direct identification of $\eta_s = 0.0908\text{ Pa}\cdot\text{s}$ (9.2% error).
4. The **Staged Pressure-Warmup Protocol** as a computationally efficient proxy.

---

## 1. The Gauge Degeneracy: Why $\eta_s$ Drifts Monotonically

### The Lack of Pressure Scale Anchoring
In the standard experimental setup, pressure is constrained only by a single Dirichlet point on the boundary:
$$p(x_0, y_0) = 0$$
This fixes the integration constant $+C$, but provides **zero boundary constraint on the amplitude/scale** of $\nabla p$.

### The Mutual Feedback Loop
Setting the derivative of the momentum loss $\mathcal{L}_{\text{mom}} = \|\mu_s^* \nabla^2 \mathbf{u} + \nabla\cdot\boldsymbol{\tau} - \nabla p\|^2$ with respect to $\mu_s^*$ to zero yields the instantaneous stationary point:
$$\mu_s^{*, \text{opt}} = \frac{\int_\Omega (\nabla p - \nabla\cdot\boldsymbol{\tau}) \cdot \nabla^2 \mathbf{u} \, d\Omega}{\int_\Omega \|\nabla^2 \mathbf{u}\|^2 \, d\Omega}$$

When `model_p` is initialized from zero at the beginning of Phase 2:
1. As the neural network learns, the amplitude and variance of $\nabla p$ grow monotonically.
2. The inner product $\int_\Omega \nabla p \cdot \nabla^2 \mathbf{u} \, d\Omega$ grows proportionally with the scale of `model_p`.
3. Consequently, $\mu_s^*$ is pulled upward.
4. An increased $\mu_s^*$ increases the target magnitude for $\nabla p$, which encourages `model_p` to inflate further.

---

## 2. The Vorticity / Curl Formulation ($\nabla \times$)

To break the mutual dependence between $\mu_s^*$ and $p(x,y)$, one applies the curl operator ($\nabla \times$) to the momentum equation. Because the curl of any gradient field is identically zero:
$$\nabla \times (\nabla p) \equiv \mathbf{0}$$

The pressure field **completely drops out of the equation**, yielding the steady viscoelastic vorticity transport equation:
$$\mu_s^* \nabla^2 \omega_z = - \nabla \times (\nabla \cdot \boldsymbol{\tau}) + Re_{\text{scale}} \nabla \times ((\mathbf{u} \cdot \nabla)\mathbf{u})$$
where $\omega_z = \frac{\partial v}{\partial x} - \frac{\partial u}{\partial y} = -\nabla^2 \psi$ is the scalar 2D vorticity.

### Empirical Validation on Four-Roll Mill (2026-08-31)
Using `train_4roll_main_curl.py` with a 5000-point sub-batch evaluation of $\mathcal{L}_{\text{curl}}$:
- **Starting guess**: $\mu_s = 0.080\text{ Pa}\cdot\text{s}$
- **Identified value**: $\mu_s = 0.0908\text{ Pa}\cdot\text{s}$ (Target COMSOL: $0.1000\text{ Pa}\cdot\text{s}$, relative error: $9.2\%$)
- **Kinematic accuracy**: $L_2(u) = 3.04\%$, $L_2(v) = 3.07\%$
- **Memory footprint**: ~3.4 GB VRAM in FP32 with chunking.

---

## 3. Practical Proxy: The Staged Phase 2 Warmup Protocol

Because evaluating 4th-order autograd derivatives during continuous training carries non-trivial GPU memory overhead, an effective and practical proxy is the **Staged Pressure-Warmup Protocol**:

1. **Phase 2A (Pressure Field Formation / Warmup)**:
   - $\mu_s$ is **frozen** at its initial guess (e.g., $0.080\text{ Pa}\cdot\text{s}$).
   - `model_p` is optimized with Adam until its spatial lobes and gradient morphology reach dynamic equilibrium with the divergence of stress $\nabla\cdot\boldsymbol{\tau}$.
2. **Phase 2B (Solvent Identification)**:
   - $\mu_s$ is **unlocked** in Adam. Because $\nabla p$ already has a stabilized scale and non-zero morphology, $\mu_s$ is optimized towards true momentum balance without triggering the initial inflation cascade.
3. **Phase 2C (High-Precision Physical Refinement)**:
   - $\mu_s$ is **frozen** at the value identified in Phase 2B.
   - L-BFGS in FP64 refines the pressure field $p(x,y)$ and cleans up the residual without parameter drift.

---

## References & Back-links
- [[Zero_Stress_BC_Compatibility]]
- [[Viscoelastic_Parameter_Identifiability]]
- [[Vorticity_Regularization]]
- [[Pressure_Stress_Decoupling]]
- [[Staged_Training_Procedure]]
- [[Viscoelastic_Training]]
