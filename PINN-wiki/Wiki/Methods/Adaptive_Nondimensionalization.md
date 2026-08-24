# Method: Adaptive Nondimensionalization

## Overview
**Adaptive Nondimensionalization** is a block-wise scaling protocol designed for inverse Viscoelastic PINNs. It prevents optimizer shortcuts and artificial degrees of freedom caused by embedding trainable physical viscosities directly into the non-dimensional governing scales (such as the Reynolds number).

Instead of treating the scaling viscosity $\eta_0$ as equal to the trainable total viscosity $\eta_{\text{tot}} = \eta_s + \eta_p$, this method separates the **numerical scaling parameter** $\eta_0$ from the **physical material parameters** $(\eta_s, \eta_p)$.

---

## Theoretical Motivation: The Run 010 Degeneracy

In classical direct PINN formulations, the Reynolds number is defined using the total physical viscosity:
$$Re = \frac{\rho U_{\text{ref}} H_{\text{ref}}}{\eta_{\text{tot}}}, \quad \text{where } \eta_{\text{tot}} = \eta_s + \eta_p$$

When $\eta_{\text{tot}}$ is treated as an active trainable variable inside the momentum equation:
$$Re(\eta_{\text{tot}}) (\mathbf{u} \cdot \nabla \mathbf{u}) + \nabla p = \frac{\eta_s}{\eta_{\text{tot}}} \nabla^2 \mathbf{u} + \nabla \cdot \boldsymbol{\tau}$$

An unconstrained optimizer discovers an artificial optimization shortcut:
$$\eta_{\text{tot}} \downarrow \quad \implies \quad Re \uparrow$$
The optimizer reduces $\eta_{\text{tot}}$ to artificially alter the relative weighting between the convection and diffusion loss terms rather than discovering the true material physics. In Run 010, this caused:
- $\beta \to 0$
- $\eta_{\text{tot}} \to 0.027\ \text{Pa}\cdot\text{s}$ (true value: $1.00\ \text{Pa}\cdot\text{s}$)
- Severe divergence of the pressure field: $L_2(p) \approx 258\%$.

---

## The Decoupled Scaling Formulation

To eliminate this spurious feedback loop, the system defines:

1. **Independent Dimensionless Viscosities**:
   $$\tilde{\eta}_s = \frac{\eta_s}{\eta_0}, \qquad \tilde{\eta}_p = \frac{\eta_p}{\eta_0}$$
2. **Fixed/Block-Frozen Scale Reynolds Number**:
   $$Re_{\text{scale}} = \frac{\rho U_{\text{ref}} H_{\text{ref}}}{\eta_0}$$
   where $\eta_0$ is **strictly not a trainable parameter** and does not backpropagate gradients.
3. **Physical Parameters Computed A Posteriori**:
   $$\eta_{\text{tot}} = \eta_s + \eta_p, \qquad \beta = \frac{\eta_s}{\eta_s + \eta_p}, \qquad Re_{\text{phys}} = \frac{\rho U_{\text{ref}} H_{\text{ref}}}{\eta_{\text{tot}}}$$
   These values are derived strictly after training and are never exposed to the loss graph.

---

## Block-Wise Adaptive Update Protocol

To ensure that the dimensionless PDE remains well-conditioned ($O(1)$ residuals) without creating rapid feedback instabilities, $\eta_0$ is updated periodically in discrete blocks rather than per-step.

```mermaid
graph TD
    A[Start Block: Freeze eta_0] --> B[Train for K=2000 Epochs via Adam/L-BFGS]
    B --> C[Estimate Total Viscosity: eta_tot = (eta_s + eta_p).detach()]
    C --> D[Compute Exponential Moving Average EMA with alpha=0.1]
    D --> E[Enforce Clamping: 0.5 * eta_0_old <= eta_0_new <= 2.0 * eta_0_old]
    E --> F[Coherently Rescale eta_s_tilde, eta_p_tilde, Re_scale]
    F --> A
```

### 1. Update Interval ($K$)
- **Phase 1 (Rheology)**: $\eta_0$ is held fixed at its arbitrary initial value (e.g., $\eta_0^{(0)} = 2.0\ \text{Pa}\cdot\text{s} \neq \eta_{\text{tot, true}}$) because $\eta_s$ is not yet trained.
- **Phase 2 (Dynamics)**: $\eta_0$ is updated every **$K = 2000$ epochs**.

### 2. Gradient Detachment & EMA Smoothing
The updated estimate of total viscosity is detached from the autograd graph:
$$\hat{\eta}_{\text{tot}} = (\eta_s + \eta_p).\text{detach}()$$
The new scaling viscosity is computed via Exponential Moving Average (EMA):
$$\eta_0^{\text{new}} = (1 - \alpha) \eta_0^{\text{old}} + \alpha \hat{\eta}_{\text{tot}}, \quad \text{with } \alpha = 0.1$$

### 3. Stability Clamping
To prevent sudden numerical shocks to the loss landscape:
$$0.5 \eta_0^{\text{old}} \le \eta_0^{\text{new}} \le 2.0 \eta_0^{\text{old}}$$

### 4. Coherent Scaling Invariance
Whenever $\eta_0$ is updated, **all** terms depending on $\eta_0$ must be rescaled simultaneously and coherently:
- $Re_{\text{scale}} \leftarrow \frac{\rho U_{\text{ref}} H_{\text{ref}}}{\eta_0^{\text{new}}}$
- $\tilde{\eta}_s \leftarrow \frac{\eta_s}{\eta_0^{\text{new}}}$
- $\tilde{\eta}_p \leftarrow \frac{\eta_p}{\eta_0^{\text{new}}}$
- Pressure and stress scaling references (if non-dimensionalized by $\eta_0$)

This guarantees that the underlying continuous physical PDE is completely invariant under the rescaling operation.

---

## Advantages
1. **Eliminates Optimization Shortcuts**: The network cannot minimize loss by simply driving $Re_{\text{scale}} \to \infty$.
2. **Separation of Time Scales**: Optimization operates at fast per-iteration time scales, while dimensional scaling operates at slow block time scales ($K=2000$).
3. **Preserves Full Blind Rigor**: Does not require initializing $\eta_0^{(0)}$ at the ground-truth total viscosity ($1.00\ \text{Pa}\cdot\text{s}$), allowing arbitrary initial values (e.g., $2.00\ \text{Pa}\cdot\text{s}$).

---

## References & Back-links
- [[Nondimensionalization]] (General dimensional scaling principles)
- [[Viscoelastic_Parameter_Identifiability]] (Analysis of parameter conditioning and Run 010 failure)
- [[Staged_Training_Procedure]] (Multi-stage execution protocol)
- [[Viscoelastic_Training]] (Implementation details in 4-Roll Mill)
