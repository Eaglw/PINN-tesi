# Topic: Viscoelastic Parameter Identifiability

## Overview
Parameter identifiability analysis for the inverse Oldroyd-B problem in complex geometries (specifically the Four-Roll Mill). In an inverse PINN setup, the goal is the **full-blind identification** of the fundamental physical parameters:
- $\lambda$: Relaxation time (s)
- $\eta_p$: Polymeric viscosity ($\text{Pa}\cdot\text{s}$)
- $\eta_s$: Solvent viscosity ($\text{Pa}\cdot\text{s}$)

From these primary parameters, composite quantities are derived **strictly a posteriori**:
$$\eta_{\text{tot}} = \eta_s + \eta_p, \qquad \beta = \frac{\eta_s}{\eta_s + \eta_p}, \qquad Re_{\text{phys}} = \frac{\rho U_{\text{ref}} H_{\text{ref}}}{\eta_{\text{tot}}}$$

The ground truth values (used solely for post-training benchmark evaluation) are:
$$\lambda = 0.05\ \text{s}, \qquad \eta_p = 0.90\ \text{Pa}\cdot\text{s}, \qquad \eta_s = 0.10\ \text{Pa}\cdot\text{s} \quad (\eta_{\text{tot}} = 1.00\ \text{Pa}\cdot\text{s}, \ \beta = 0.10)$$

---

## Mathematical Parameterization: Log-Space Representation

To guarantee physical admissibility without gradient distortion, parameters are optimized in logarithmic space:
$$\lambda = \lambda_{\text{ref}} e^{r_\lambda}, \qquad \eta_p = \eta_{p,\text{ref}} e^{r_p}, \qquad \eta_s = \eta_{s,\text{ref}} e^{r_s}$$
where $r_\lambda, r_p, r_s$ are the unconstrained trainable weights.

### Advantages:
1. **Strict Positivity**: Enforces $\lambda > 0, \eta_p > 0, \eta_s > 0$ unconditionally without clipping or projection artifacts.
2. **Scale Invariance**: Equalizes relative gradient steps across parameters spanning different physical magnitudes.
3. **Avoids Saturation**: Eliminates the gradient flattening observed with `softplus` activation near zero.
4. **Decoupled Formulation**: Eliminates the artificial algebraic coupling $\beta + (1 - \beta) = 1$ as a primary constraint.

> [!IMPORTANT]
> **Full-Blind Requirement**: The numerical references ($\lambda_{\text{ref}}, \eta_{p,\text{ref}}, \eta_{s,\text{ref}}$) must be chosen purely as arbitrary numerical scale factors (e.g., $1.0, 1.0, 1.0$) and must **never** encode ground-truth material priors.

---

## Offline Identifiability Tests (COMSOL High-Fidelity Data)

To determine whether the physical flow field inherently contains sufficient information to decouple $\lambda, \eta_p$, and $\eta_s$, rigorous offline least-squares and Singular Value Decomposition (SVD) tests were conducted directly on exact numerical simulation fields.

### 1. Rheological Identifiability & SVD Conditioning
The dimensionless Oldroyd-B constitutive equation relates the stress tensor $\boldsymbol{\tau}^*$ to the deformation tensor $\mathbf{D}^*$:
$$\boldsymbol{\tau}^* + Wi \overset{\triangledown}{\boldsymbol{\tau}^*} = 2 \tilde{\eta}_p \mathbf{D}^*$$
where $Wi = \lambda \frac{U_{\text{ref}}}{H_{\text{ref}}}$ and $\tilde{\eta}_p = \frac{\eta_p}{\eta_0}$.

The sensitivity matrix $J_{\text{con}}$ evaluated across all spatial collocation points decomposes into two orthogonal directions associated with the upper-convected derivative $\overset{\triangledown}{\boldsymbol{\tau}}$ and the deformation rate tensor $\mathbf{D}$:
- Singular values: $\sigma_1 = 385.1, \quad \sigma_2 = 286.5$
- Condition number:
  $$\kappa(J_{\text{con}}) = \frac{\sigma_1}{\sigma_2} = \mathbf{1.34}$$

**Least-Squares Recovery Results**:
- $\lambda = 0.049950\ \text{s} \quad (\mathbf{0.10\%} \text{ relative error})$
- $\eta_p = 0.900062\ \text{Pa}\cdot\text{s} \quad (\mathbf{0.01\%} \text{ relative error})$

> [!NOTE]
> **Conclusion**: The rheological system $(\lambda, \eta_p)$ exhibits near-ideal conditioning ($\kappa \approx 1.34$). The four-roll mill kinematics provide rich elongational and shear gradients that strongly decouple relaxation time from polymeric viscosity.

---

### 2. Solvent Viscosity Identification: Direct Momentum vs. Curl-Momentum
Two distinct formulations for isolating solvent viscosity $\eta_s$ were tested on the momentum balance:

#### A. Curl-Momentum Formulation ($\nabla \times \nabla p \equiv 0$)
Eliminating pressure by taking the curl of the momentum equation:
$$\nabla \times \left[ Re_{\text{scale}} (\mathbf{u} \cdot \nabla \mathbf{u}) - \tilde{\eta}_s \nabla^2 \mathbf{u} - \nabla \cdot \boldsymbol{\tau} \right] = 0$$
- Result: $\eta_s \approx 0.0051\ \text{Pa}\cdot\text{s} \quad (\mathbf{95\%} \text{ relative error})$.
- **Root Cause**: High-order finite differences amplify discretization noise. For grid spacing $\Delta x \approx 0.028$, the 2nd derivative operator scales as $\frac{1}{\Delta x^2} \approx 1250$, and the 3rd/4th order derivatives in the curl operator scale as $\frac{1}{\Delta x^3} \approx 45000$, destroying the signal.

#### B. Direct Momentum Formulation
Directly fitting the momentum balance with pressure gradients:
$$\nabla p + Re_{\text{scale}} (\mathbf{u} \cdot \nabla \mathbf{u}) - \nabla \cdot \boldsymbol{\tau} = \tilde{\eta}_s \nabla^2 \mathbf{u}$$
- Spatial correlation: **$0.8929$**
- Result: $\eta_s = 0.098971\ \text{Pa}\cdot\text{s} \quad (\mathbf{1.03\%} \text{ relative error})$.

**Full Noise-Free System Reconstruction**:
- $\eta_{\text{tot}} = 0.999033\ \text{Pa}\cdot\text{s} \quad (\mathbf{0.10\%} \text{ error})$
- $\beta = 0.099067 \quad (\mathbf{0.93\%} \text{ error})$

---

## Synthetic Noise Robustness & Finite Differences vs. PINN Autodiff

Gaussian noise ($\mathcal{N}(0, \sigma^2)$) was added to the observational fields $(u, v, p, \boldsymbol{\tau})$ to evaluate estimation stability:

| Noise Level | Condition Number $\kappa(J_{\text{con}})$ | $\lambda$ Error | $\eta_p$ Error | $\eta_s$ Error (Finite Diff.) |
| :---: | :---: | :---: | :---: | :---: |
| **0.0%** | 1.34 | 0.10% | 0.01% | 1.03% |
| **0.1%** | 1.34 | 0.18% | 0.06% | 94.5% |
| **0.5%** | 1.34 | 2.16% | 1.70% | 99.7% |
| **1.0%** | 1.33 | 7.31% | 6.49% | 99.7% |
| **2.0%** | 1.29 | 25.3% | 21.8% | 100.0% |

### Key Physical & Algorithmic Takeaways:
1. **Rheological Robustness**: The constitutive inversion $(\lambda, \eta_p)$ remains remarkably stable even under $1\%$ noise ($<7.5\%$ error), confirming that physical identifiability is structurally sound and not a numerical artifact of pristine data.
2. **Finite Difference Breakdown vs. PINN Autograd Advantage**:
   - The collapse of $\eta_s$ under noise is an artifact of **discrete numerical differentiation** (where finite differences act as noise amplifiers), not an intrinsic failure of momentum physics.
   - **PINN Advantage**: The neural network acts as a smooth, continuous global approximator. Derivatives are evaluated via **exact analytical automatic differentiation (Autograd)** on the learned neural manifold, filtering high-frequency noise and preserving the viscous Laplacian signal $\nabla^2 \mathbf{u}$.

---

## The Run 010 Failure Mode & Solutions

In previous iterations (Run 010), coupling $Re = \frac{\rho U H}{\eta_{\text{tot}}}$ with trainable $\eta_{\text{tot}}$ led to an optimization shortcut:
$$\eta_{\text{tot}} \downarrow \quad \implies \quad Re \uparrow \quad \implies \quad \beta \to 0, \quad \eta_{\text{tot}} \to 0.027\ \text{Pa}\cdot\text{s}, \quad L_2(p) \approx 258\%$$

### Structural Countermeasures:
1. **Scale Decoupling**: Scale Reynolds $Re_{\text{scale}} = \frac{\rho U H}{\eta_0}$ is fixed/frozen per training block via [[Adaptive_Nondimensionalization]].
2. **Decoupled Two-Phase Optimization**: Phase 1 identifies $(\lambda, \eta_p)$ with frozen pressure; Phase 2 identifies $\eta_s$ with frozen stress and soft velocity micro-adjustments via [[Soft_Anti_Drift]].
3. **Multi-Start Verification**: Testing across multiple random initializations spanning orders of magnitude to confirm global basin of attraction.

---

## References & Back-links
- [[Adaptive_Nondimensionalization]] (Block-wise scaling protocol)
- [[Soft_Anti_Drift]] (Kinematic stabilization during momentum training)
- [[Staged_Training_Procedure]] (Multi-stage training workflow)
- [[Pressure_Stress_Decoupling]] (Helmholtz-Hodge decomposition and pressure isolation)
- [[Viscoelastic_Training]] (System experiment configuration)
