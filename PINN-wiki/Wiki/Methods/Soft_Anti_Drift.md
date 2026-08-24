# Method: Soft Anti-Drift Regularization

## Overview
**Soft Anti-Drift Regularization** is a stabilization technique introduced in the Phase 2 training of Viscoelastic PINNs. It resolves the fundamental conflict between the **Helmholtz-Hodge pressure inference limit** (which causes severe pressure divergence under a rigid velocity freeze) and unconstrained velocity drift (which corrupts the kinematic profile discovered in Phase 1).

Instead of a rigid *hard freeze* or an unconstrained unfreezing of the stream function network `model_psi`, Soft Anti-Drift unlocks $\psi$ with a dedicated low learning rate ($LR_\psi \approx 10^{-4}$) while penalizing deviations from the Phase 1 velocity checkpoint.

---

## Mathematical Formulation

The soft anti-drift loss $\mathcal{L}_{\text{drift}}$ is defined as the normalized mean squared deviation of the velocity field $\mathbf{u} = (u, v)$ from the checkpoint velocity $\mathbf{u}_{\text{checkpoint}}$ saved at the end of Phase 1:

$$\mathcal{L}_{\text{drift}} = \frac{\|\mathbf{u} - \mathbf{u}_{\text{checkpoint}}\|^2}{\|\mathbf{u}_{\text{checkpoint}}\|^2 + \epsilon} = \frac{\frac{1}{N}\sum_{i=1}^N \left( (u^{(i)} - u_{\text{ckpt}}^{(i)})^2 + (v^{(i)} - v_{\text{ckpt}}^{(i)})^2 \right)}{\frac{1}{N}\sum_{i=1}^N \left( (u_{\text{ckpt}}^{(i)})^2 + (v_{\text{ckpt}}^{(i)})^2 \right) + \epsilon}$$

where $\epsilon = 10^{-6}$ prevents numerical division by zero.

---

## Physical Rationale: Overcoming the Helmholtz-Hodge Bottleneck

In Phase 2, the momentum equation governs the pressure gradient:
$$\nabla p = - Re_{\text{scale}} (\mathbf{u} \cdot \nabla \mathbf{u}) + \tilde{\eta}_s \nabla^2 \mathbf{u} + \nabla \cdot \boldsymbol{\tau} \equiv \mathbf{f}(\mathbf{u}, \boldsymbol{\tau})$$

By the Helmholtz-Hodge theorem, any vector field $\mathbf{f}$ decomposes into a curl-free (conservative) component and a divergence-free (solenoidal) component:
$$\mathbf{f} = \nabla p_{\text{true}} + \mathbf{g}, \quad \text{with } \nabla \cdot \mathbf{g} = 0$$

Because $\nabla \times \nabla p \equiv 0$, the pressure network `model_p` can **only** fit the conservative part $\nabla p_{\text{true}}$. It is mathematically incapable of absorbing any non-zero rotational component $\mathbf{g}$ ($\nabla \times \mathbf{f} \neq 0$).

### 1. Why Hard Freeze Fails
When $\psi$ and $\boldsymbol{\tau}$ are frozen rigidly from Phase 1:
- The Laplacian term $\nabla^2 \mathbf{u}$ requires computing 3rd-order spatial derivatives of $\psi$.
- Tiny, smooth approximation errors in $\psi$ ($\sim 0.5\% - 1\%$ L2 error) are amplified dramatically by high-order differentiation, creating a large spurious solenoidal force $\mathbf{g}$.
- `model_p` cannot balance $\mathbf{g}$, causing the momentum loss to hit an artificial plateau and the pressure field to diverge catastrophically (as observed in Run 010 where $L_2(p) \approx 258\%$).

### 2. Why Unconstrained Unfreezing Fails
If `model_psi` is completely unfrozen with standard learning rates, the momentum loss gradients overwhelm the boundary and data constraints, causing the stream function to distort and "drift" far from the true physical kinematics.

### 3. The Soft Constraint Balance
Soft Anti-Drift provides the exact degrees of freedom needed:
$$\text{Hard Freeze (Rigid)} \quad \longrightarrow \quad \mathbf{\text{Soft Anti-Drift (Micro-Adjustments)}} \quad \longleftarrow \quad \text{Unconstrained Drift}$$
The network is penalized if it alters macroscopic flow streamlines, but possesses sufficient flexibility at low learning rates ($LR = 10^{-4}$) to make the minute, high-frequency spatial corrections ($\Delta u / u \ll 0.1\%$) required to eliminate the rotational residual $\mathbf{g} \to 0$.

---

## Training Hyperparameters & Scheduling

During Phase 2 (Dynamics & Solvent Viscosity identification):
- `model_p`: Active ($LR_p = 10^{-3}$)
- `model_psi`: Active with Soft Anti-Drift ($LR_\psi = 10^{-4}$)
- `model_tau`: Rigidly Frozen
- Trainable Parameter $\eta_s$: Active ($LR_{\eta_s} = 10^{-4}$ in log-space)
- Frozen Parameters: $\lambda, \eta_p$

The total Phase 2 optimization loss is:
$$\mathcal{L}_{\text{Phase 2}} = \mathcal{L}_{\text{momentum}} + \lambda_u \mathcal{L}_u + \lambda_{\text{anchor}} \mathcal{L}_{p,\text{anchor}} + \lambda_{\text{drift}} \mathcal{L}_{\text{drift}}$$

---

## Diagnostic Metrics to Monitor
To verify that Soft Anti-Drift is functioning correctly without hidden parameter compensations:
1. **Kinematic Drift Ratio**:
   $$\text{Drift}_u = \frac{\|\mathbf{u} - \mathbf{u}_{\text{ckpt}}\|}{\|\mathbf{u}_{\text{ckpt}}\|}$$
   *Target: Should remain $< 0.5\%$.*
2. **Curvature Drift Ratio**:
   $$\text{Drift}_{\nabla^2 u} = \frac{\|\nabla^2 \mathbf{u} - \nabla^2 \mathbf{u}_{\text{ckpt}}\|}{\|\nabla^2 \mathbf{u}_{\text{ckpt}}\| + \epsilon}$$
   *Monitors whether $\eta_s$ is artificially compensating for shifts in fluid viscous diffusion.*

---

## References & Back-links
- [[Pressure_Stress_Decoupling]] (Helmholtz-Hodge decomposition theory)
- [[Staged_Training_Procedure]] (Two-phase training orchestration)
- [[Viscoelastic_Parameter_Identifiability]] (Solvent viscosity identification)
- [[Viscoelastic_Training]] (System training manual)
