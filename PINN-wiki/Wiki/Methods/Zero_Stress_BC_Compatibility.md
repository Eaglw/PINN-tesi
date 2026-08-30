# Method: Zero-Stress BC Compatibility in Phase 1 (Full-PIV Rheometry)

## Overview

In standard viscoelastic PINN training for closed recirculating flows (such as the four-roll mill), Phase 1 typically enforces:
1. Kinematic velocity supervision ($\mathbf{u}, \mathbf{v}$) in the domain $\Omega$ and on boundaries $\Gamma_{\text{walls}}, \Gamma_{\text{rolls}}$.
2. The constitutive PDE residual $\boldsymbol{\tau} + \lambda \stackrel{\triangledown}{\boldsymbol{\tau}} = 2 \mu_p \mathbf{D}(\mathbf{u})$.
3. **Dirichlet Boundary Conditions for Extra-Stress $\boldsymbol{\tau}$ on the 4 rolls** ($\Gamma_{\text{rolls}}$), extracted from numerical CFD (COMSOL).

While effective, requiring $\boldsymbol{\tau}$ on the rolls limits the method's applicability to **real experimental PIV (Particle Image Velocimetry) data**, where stress cannot be measured optically.

This page documents the **Zero-Stress-BC Formulation**, which replaces artificial roll-stress Dirichlet data with the **Momentum Curl Compatibility Constraint** in Phase 1:
$$\text{curl}(\mathbf{F}) = \nabla \times \left( \nabla \cdot \boldsymbol{\tau} + \mu_s \nabla^2 \mathbf{u} - Re (\mathbf{u}\cdot\nabla)\mathbf{u} \right) \equiv 0$$

---

## 1. Physical & Mathematical Foundations

### The Hyperbolic Dilemma of Closed Streamlines
The Oldroyd-B constitutive equation is convective-hyperbolic along streamlines:
$$(\mathbf{u} \cdot \nabla) \boldsymbol{\tau} = \frac{1}{\lambda} \left( 2 \mu_p \mathbf{D} - \boldsymbol{\tau} \right) + (\nabla \mathbf{u}) \boldsymbol{\tau} + \boldsymbol{\tau} (\nabla \mathbf{u})^T$$

In open channel flows, stress boundary conditions at the inflow anchor the stress manifold. In a closed four-roll mill, streamlines form closed concentric orbits. Without an absolute stress boundary condition:
- The constitutive equation determines $\boldsymbol{\tau}$ relative to deformation rates $\mathbf{D}$, but the absolute scale of $\mu_p$ and $\boldsymbol{\tau}$ can be degenerate if velocity data alone is used with $w_{\text{mom}} = 0$.

### The Momentum Compatibility Anchor
By Helmholtz's theorem, a vector force field $\mathbf{F}(\mathbf{x})$ can balance a scalar pressure gradient $\nabla p$ if and only if its rotational part is identically zero:
$$\nabla \times \mathbf{F} = \mathbf{0} \iff \exists p \text{ s.t. } \nabla p = \mathbf{F}$$

Expanding $\mathbf{F} = \nabla \cdot \boldsymbol{\tau} + \mu_s \nabla^2 \mathbf{u} - Re(\mathbf{u}\cdot\nabla)\mathbf{u}$:
$$\text{curl}(\nabla \cdot \boldsymbol{\tau}) = - \mu_s \text{curl}(\nabla^2 \mathbf{u}) + Re \, \text{curl}((\mathbf{u}\cdot\nabla)\mathbf{u})$$

### Why This Replaces Roll-Stress BCs:
1. **Direct Scale Anchoring**: The kinematic field $\mathbf{u}$ is strictly supervised by PIV/COMSOL data with a known physical velocity scale $U_{\text{ref}}$. Therefore, $-\mu_s \text{curl}(\nabla^2 \mathbf{u})$ has a **fixed, absolute physical magnitude**.
2. **Coupling Without Pressure**: The curl operator completely eliminates the pressure field $p(x,y)$. The stress network $\text{model\_tau}$ is forced to generate an extra-stress field $\boldsymbol{\tau}$ whose spatial divergence $\nabla \cdot \boldsymbol{\tau}$ has the exact rotational magnitude needed to balance the observed kinematics.
3. **100% Unsupervised Stress**: Zero stress measurements are required anywhere in the domain or on the boundaries ($\Gamma_{\text{walls}}, \Gamma_{\text{rolls}}$).

---

## 2. Implementation Architecture in Phase 1

### Loss Function
In Phase 1 with Zero-Stress BCs:
$$\mathcal{L}_{\text{Phase 1}} = W_{\text{data}} \mathcal{L}_{\text{uv}} + W_{\text{bc}} \mathcal{L}_{\text{bc,vel}} + W_{\text{con}} \mathcal{L}_{\text{constitutive}} + W_{\text{curl}} \mathcal{L}_{\text{curl}}$$
where:
$$\mathcal{L}_{\text{curl}} = \frac{1}{N_{\text{sub}}} \sum_{i=1}^{N_{\text{sub}}} \left| \frac{\partial F_y}{\partial x} - \frac{\partial F_x}{\partial y} \right|^2$$

### Computational Strategy (VRAM & Autograd Management)
- Because $\text{curl}(\nabla^2 \mathbf{u})$ involves 4th-order spatial derivatives of $\psi$ and $\text{curl}(\nabla \cdot \boldsymbol{\tau})$ involves 2nd-order derivatives of $\boldsymbol{\tau}$, evaluating $\mathcal{L}_{\text{curl}}$ over all 125,000 collocation points would exceed GPU memory.
- **Subsampling Protocol**: $\mathcal{L}_{\text{curl}}$ is evaluated on a random mini-batch of $N_{\text{sub}} \in [3000, 5000]$ internal collocation points per step.
- **Memory Footprint**: Measured at ~3.4 GB VRAM in FP32 on NVIDIA GPUs, well within standard hardware limits (12 GB target).

---

## 3. Comparison of Paradigms

| Feature | Classical Phase 1 | Zero-Stress-BC (Curl Compatibility) |
| :--- | :--- | :--- |
| **Stress Supervision** | 8,000 COMSOL points on rolls ($\Gamma_{\text{rolls}}$) | **0 points (100% unsupervised)** |
| **Experimental Feasibility** | Requires synthetic CFD stress | **Fully compatible with experimental PIV** |
| **Governing Loss** | $\mathcal{L}_{\text{uv}} + \mathcal{L}_{\text{bc}} + \mathcal{L}_{\text{con}}$ | $\mathcal{L}_{\text{uv}} + \mathcal{L}_{\text{bc}} + \mathcal{L}_{\text{con}} + W_{\text{curl}}\mathcal{L}_{\text{curl}}$ |
| **Stress-Kinematics Coupling** | Decoupled from momentum | **Coupled via irrotational force balance** |
| **Autograd Derivative Order** | Order 2 ($\nabla \mathbf{u}, \nabla \boldsymbol{\tau}$) | Order 4 for $\psi$, Order 2 for $\boldsymbol{\tau}$ |

---

## References & Back-links
- [[Vorticity_Inversion_Solvent]]
- [[Pressure_Stress_Decoupling]]
- [[Viscoelastic_Parameter_Identifiability]]
- [[ViscoelasticNet_Full model]]
- [[Viscoelastic_Training]]
