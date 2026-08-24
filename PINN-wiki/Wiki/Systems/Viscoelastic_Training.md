# System: Viscoelastic Training Experiment

## Overview
This document serves as the comprehensive technical, architectural, and operational specification for the Viscoelastic PINN inverse solver applied to the Four-Roll Mill geometry (`final_roll/train_4roll_main.py` and `final_roll/src/`). It documents the full-blind parameter identification framework for recovering Oldroyd-B physical properties ($\lambda, \eta_p, \eta_s$) without prior material knowledge.

---

## Neural Network Architecture

The solver decouples the discovery of physical fields using dedicated neural network heads under the `CombinedModel` container:

### 1. Sub-Network Definitions
- **Stream Function (`model_psi`)**: Fully Connected Network taking coordinates $(x, y) \in [0, 1]^2$ and predicting scalar $\psi$. Velocity components are computed via exact automatic differentiation:
  $$u = \frac{\partial \psi}{\partial y}, \qquad v = -\frac{\partial \psi}{\partial x}$$
  This automatically guarantees the incompressibility constraint ($\nabla \cdot \mathbf{u} = 0$) by construction.
- **Pressure (`model_p`)**: FCN predicting scalar pressure field $p(x, y)$.
- **Polymeric Stress (`model_tau`)**: FCN predicting the three independent components of the symmetric 2D extra-stress tensor: $\boldsymbol{\tau} = (\tau_{xx}, \tau_{xy}, \tau_{yy})$.

### 2. Architectural Conventions
- **Activation Function**: `nn.SiLU` ([[Activation_Functions]]) across all hidden layers due to its smooth second derivatives required for stable momentum Laplacians ($\nabla^2 \mathbf{u}$).
- **Explicit Zero Initialization**: The final layers of `model_tau` and `model_p` are initialized to zero (`initialize_last_layer_zero()`) to prevent random initialization noise from destabilizing early kinematic learning.
- **Logarithmic Parameter Space**: Physical parameters are optimized in unconstrained log-space:
  $$\lambda = \lambda_{\text{ref}} e^{r_\lambda}, \qquad \eta_p = \eta_{p,\text{ref}} e^{r_p}, \qquad \eta_s = \eta_{s,\text{ref}} e^{r_s}$$
  Numerical references are arbitrary scaling factors (e.g., $1.0$) with no prior material knowledge.

---

## Two-Phase Decoupled Training Orchestration

Training is strictly divided into two distinct physical phases. Joint coupled training (Phase 3) is **deprecated** as it destabilizes stress field learning.

```mermaid
graph TD
    A[Phase 1: Rheology & Kinematics<br>Adam FP32: 20k ep + L-BFGS FP64: 5k st<br>Active: psi, tau, r_lambda, r_p | Frozen: p, r_s<br>Loss: Constitutive + BCs u, tau_roll<br>eta_0 = 2.0 Pa s fixed] --> B[Phase 2: Hydrodynamics & Solvent Viscosity<br>Adam FP32: 15k ep + L-BFGS FP64: 5k st<br>Active: p, r_s, psi low-LR | Frozen: tau, r_lambda, r_p<br>Loss: Momentum + Drift + BCs u, p_anchor<br>Adaptive eta_0 update every 2000 ep]
    B --> C[Post-Training Evaluation & Reconstruction<br>Compute a posteriori: eta_tot, beta, Re_phys<br>Compare with Ground Truth solely for benchmark metrics]
```

### Phase 1: Kinematics & Rheology
- **Active Networks**: `model_psi`, `model_tau`
- **Frozen Networks**: `model_p` (frozen at zero)
- **Trainable Parameters**: $r_\lambda, r_p$ (recovering relaxation time $\lambda$ and polymeric viscosity $\eta_p$)
- **Frozen Parameters**: $r_s$ (solvent viscosity $\eta_s$)
- **Active Loss**:
  $$\mathcal{L}_{\text{Phase 1}} = \mathcal{L}_{\text{constitutive}} + \lambda_u \mathcal{L}_{u,\text{data}} + \lambda_{\text{roll}} \mathcal{L}_{\boldsymbol{\tau},\text{roll}}$$
- **Optimization Strategy**:
  1. **Adam @ FP32** (20,000 epochs, $LR = 10^{-3}$) for broad convex basin discovery.
  2. **L-BFGS @ FP64** (~5,000 steps) for high-precision convergence of $(\lambda, \eta_p)$ and stress field topology.

### Phase 2: Hydrodynamics & Solvent Viscosity
- **Active Networks**: `model_p` ($LR_p = 10^{-3}$), `model_psi` ($LR_\psi = 10^{-4}$ with soft anti-drift)
- **Frozen Networks**: `model_tau` (rigidly frozen)
- **Trainable Parameters**: $r_s$ ($LR_{\eta_s} = 10^{-4}$)
- **Frozen Parameters**: $r_\lambda, r_p$ (frozen to prevent constitutive corruption)
- **Active Loss**:
  $$\mathcal{L}_{\text{Phase 2}} = \mathcal{L}_{\text{momentum}} + \lambda_u \mathcal{L}_{u} + \lambda_{\text{anchor}} \mathcal{L}_{p,\text{anchor}} + \lambda_{\text{drift}} \mathcal{L}_{\text{drift}}$$
  where [[Soft_Anti_Drift]] loss $\mathcal{L}_{\text{drift}} = \frac{\|\mathbf{u} - \mathbf{u}_{\text{ckpt}}\|^2}{\|\mathbf{u}_{\text{ckpt}}\|^2 + \epsilon}$ overcomes the [[Pressure_Stress_Decoupling#The Helmholtz-Hodge Pressure Inference Limit|Helmholtz-Hodge limit]].
- **Adaptive Nondimensionalization**: Every $K = 2000$ epochs, $\eta_0$ is updated via detached EMA ($\alpha=0.1$, clamping $[0.5, 2.0]$) via [[Adaptive_Nondimensionalization]].
- **Optimization Strategy**:
  1. **Adam @ FP32** (15,000 epochs).
  2. **L-BFGS @ FP64** (~5,000 steps) for definitive scientific-grade convergence.

---

## The Run 010 Autopsy & Structural Countermeasures

| Feature | Legacy Setup (Run 010 Failure) | Modern Decoupled Paradigm |
| :--- | :--- | :--- |
| **Reynolds Definition** | $Re = \frac{\rho U H}{\eta_{\text{tot}}}$ (Trainable $\eta_{\text{tot}}$) | $Re_{\text{scale}} = \frac{\rho U H}{\eta_0}$ (Fixed numerical scale $\eta_0$) |
| **Viscosity Primary Variables** | $\beta = \frac{\eta_s}{\eta_{\text{tot}}}$ and $\eta_{\text{tot}}$ | Independent $\eta_s > 0, \eta_p > 0$ in log-space |
| **Phase 2 Velocity Constraint** | Hard Freeze of `model_psi` | Soft Anti-Drift $\mathcal{L}_{\text{drift}}$ with $LR_\psi = 10^{-4}$ |
| **Coupled Refinement** | Phase 3 unfreezing all networks | Phase 3 **Deprecated**; decoupled 2-stage L-BFGS |
| **Result** | $\beta \to 0, \eta_{\text{tot}} \to 0.027, L_2(p) \approx 258\%$ | $\lambda$ err $0.10\%, \eta_p$ err $0.01\%, \eta_s$ err $1.03\%$ (offline) |

---

## Diagnostic Monitoring & Evaluation Protocol

During training and post-processing, the following state variables must be tracked simultaneously:

1. **Physical Parameter Trajectories**:
   $$\lambda(t), \quad \eta_p(t), \quad \eta_s(t), \quad \eta_{\text{tot}}(t) = \eta_s(t) + \eta_p(t), \quad \beta(t) = \frac{\eta_s(t)}{\eta_{\text{tot}}(t)}$$
2. **Scaling & Dimensionless Trajectories**:
   $$\eta_0(t), \quad Re_{\text{scale}}(t) = \frac{\rho U H}{\eta_0(t)}, \quad Re_{\text{phys}}(t) = \frac{\rho U H}{\eta_{\text{tot}}(t)}$$
3. **Loss Residuals**:
   $$\mathcal{L}_{\text{constitutive}}(t), \quad \mathcal{L}_{\text{momentum}}(t), \quad \mathcal{L}_{\text{drift}}(t)$$
4. **Kinematic & Curvature Stability Ratios**:
   $$\text{Drift}_u = \frac{\|\mathbf{u} - \mathbf{u}_{\text{ckpt}}\|}{\|\mathbf{u}_{\text{ckpt}}\|}, \qquad \text{Drift}_{\nabla^2 u} = \frac{\|\nabla^2 \mathbf{u} - \nabla^2 \mathbf{u}_{\text{ckpt}}\|}{\|\nabla^2 \mathbf{u}_{\text{ckpt}}\| + \epsilon}$$

---

## Experimental Validation Roadmap (Full-Blind Claim)

To rigorously substantiate the claim of full-blind parameter discovery, the following experimental sequence is established:

1. **Test 1 — Single Blind Run**:
   - Initial scaling scale: $\eta_0^{(0)} = 2.0\ \text{Pa}\cdot\text{s}$ (deliberately offset from ground-truth $1.0\ \text{Pa}\cdot\text{s}$).
   - Parameter initializations chosen without material priors.
2. **Test 2 — Multi-Start Basin of Attraction**:
   - Repeat inversion across multiple initial guesses $(\lambda^{(0)}, \eta_p^{(0)}, \eta_s^{(0)})$ distributed across orders of magnitude.
   - Verify convergence to the identical global attractor basin.
3. **Test 3 — Normalization Scale Invariance**:
   - Test varying arbitrary initial scales: $\eta_0^{(0)} \in \{0.5, 2.0, 5.0\}\ \text{Pa}\cdot\text{s}$.
   - Verify that final physical parameters $(\lambda, \eta_p, \eta_s)$ are independent of numerical scale $\eta_0$.
4. **Test 4 — Noise-Aware Full PINN Training**:
   - Add $0.5\%$ and $1.0\%$ Gaussian noise to observational velocity/stress data during complete PINN training (validating the smooth Autograd hypothesis over finite differences).
5. **Test 5 — Ablation Study**:
   - Quantify contributions by benchmarking:
     1. $\psi$ Hard Freeze vs. $\psi$ Unconstrained vs. $\psi$ Soft Anti-Drift.
     2. Static $\eta_0$ vs. Block-wise Adaptive $\eta_0$.

---

## Related Wiki Links
- **Theory & Physics**: [[Viscoelastic_Fluids]], [[Viscoelastic_Parameter_Identifiability]], [[Pressure_Stress_Decoupling]], [[Nondimensionalization]]
- **Methods**: [[Soft_Anti_Drift]], [[Adaptive_Nondimensionalization]], [[Staged_Training_Procedure]], [[Staged_Precision_Strategy]], [[ViscoelasticNet]]
