# Method: Viscoelastic Mesh Convergence and Worst-Case Boundary Protocol

## Overview
The **Viscoelastic Mesh Convergence Protocol** defines the rigorous numerical methodology used to prove the **spatial grid independence** of COMSOL Multiphysics reference solutions in the **Four-Roll Mill**, and governs how these validated spatial resolutions are transferred to Physics-Informed Neural Network (PINN) training datasets.

In numerical rheology, conducting a comprehensive mesh convergence study across every permutation of rheological parameters ($\lambda, \eta_p, \eta_s, \alpha, \varepsilon$) is computationally prohibitive and scientifically redundant. Instead, this protocol establishes the **Worst-Case Limiting Principle (Principio del Caso Limite)**: by rigorously demonstrating grid convergence on the mathematically most singular and demanding configuration, that mesh resolution is formally proven sufficient for all less restrictive flow regimes and non-linear constitutive models.

---

## 1. Theoretical Foundation: The "Worst-Case" Principle

As established in [[High_Weissenberg_Number_Problem]], the spatial gradient severity in viscoelastic flows is governed by:
1. **The Weissenberg Number ($Wi = \lambda \frac{U_{\text{ref}}}{H_{\text{ref}}}$)**: Higher relaxation times $\lambda$ produce exponentially steep stress wakes and thin boundary layers ($\delta \propto Wi^{-1}$).
2. **Constitutive Regularization**:
   - **Oldroyd-B** lacks any stress-saturation mechanism ($\eta_E \to \infty$ at $\dot{\varepsilon} \to 1/(2\lambda)$), making it the most singular and mesh-sensitive model in existence.
   - **Giesekus** ($\alpha > 0$, $\delta \propto Wi^{-1/2}$) and **PTT** ($\varepsilon > 0$, $\delta \propto Wi^{-1/3}$) feature non-linear shear-thinning and bounded extensional viscosities, yielding inherently broader and more regular boundary layers.

```
       RIGOROUS GRID INDEPENDENCE TRANSFER (THE "APPEND" STRATEGY)

   [ Oldroyd-B at λ_max (e.g., 0.1 s - 0.2 s) ]
   Tested on: M1 (125k) -> M2 (88k) -> M3 (52k) -> M4 (25k)
                      │
                      ▼
   [ Relative L2 Difference < 0.5% - 1.0% ]
   Convergence Threshold Identified (e.g., M3 / 52k nodes)
                      │
                      ├──────────────────────────────────────────────┐
                      ▼                                              ▼
   [ Lower Elasticity Regimes ]                    [ Regularized Constitutive Models ]
   • Oldroyd-B (λ = 0.05 s)                        • Giesekus (α = 0.1 - 0.3)
   • Thicker boundary layers                       • PTT (ε = 0.1 - 0.25)
   • Smooth, benign gradients                      • Bounded extensional stress
   ==> INHERITS MESH AUTOMATICALLY                 ==> INHERITS MESH AUTOMATICALLY
       (Zero redundant CFD sweeps)                     (Zero redundant CFD sweeps)
```

---

## 2. COMSOL Discretization Levels

Four spatial discretization levels are evaluated for the 2D Four-Roll Mill geometry ($L \times H = 0.10\text{ m} \times 0.10\text{ m}$, roll radii $R = 0.015\text{ m}$):

| Mesh Level | COMSOL Preset | Approximate Node Count | Purpose in Convergence Study |
|:---|:---|:---|:---|
| **M1** | *Extremely Fine* | $\approx 125.000$ | Asymptotic reference benchmark ("truth") |
| **M2** | *Extra Fine* | $\approx 88.000$ | Intermediate high-resolution verification |
| **M3** | *Finer* | $\approx 52.000$ | Target production resolution |
| **M4** | *Fine* | $\approx 25.000 - 30.000$ | Coarse lower bound (identifies numerical breakdown knee) |

> [!IMPORTANT]
> **Lower Bound Constraint**: Meshes coarser than $\approx 15.000 - 20.000$ nodes fail to adequately discretize the narrow pinch gap between adjacent counter-rotating cylinders ($d_{\text{gap}} \approx 0.01\text{ m}$), leading to artificial streamline diffusion (DEVSS/GLS numerical dissipation) and under-predicting the peak extensional stress at the central saddle point $(0, 0)$ by $> 15\%$.

---

## 3. Quantitative Verification Protocol (Cut-lines & Metrics)

To prove grid independence without relying solely on solver residuals, three specific diagnostics must be evaluated across $M_k$:

### Diagnostic 1: Critical Shear Velocity Profile in Roll Gap
Extract velocity $u(y)$ along the vertical cut-line bisecting the upper roll gap ($x = 0.025\text{ m}, y \in [0.01, 0.04]\text{ m}$):
$$
E_{L_2}(u; M_k) = \frac{\|u_{M_k}(y) - u_{M_1}(y)\|_{L_2}}{\|u_{M_1}(y)\|_{L_2}}
$$
- **Criterion**: $E_{L_2}(u; M_k) < 0.5\%$.

### Diagnostic 2: Peak Extensional Stress at Central Saddle Point
Extract $\tau_{xx}(x)$ along the central horizontal outflow streamline ($y = 0, x \in [-0.02, 0.02]\text{ m}$):
$$
E_{\text{peak}}(\tau_{xx}; M_k) = \frac{|\tau_{xx, M_k}(0, 0) - \tau_{xx, M_1}(0, 0)|}{|\tau_{xx, M_1}(0, 0)|}
$$
- **Criterion**: $E_{\text{peak}}(\tau_{xx}; M_k) < 1.5\%$.

### Diagnostic 3: Global Grid Convergence Index (GCI)
Based on Roache's verification standard:
$$
\text{GCI}_{23} = \frac{F_s |\varepsilon_{23}|}{r^p - 1}
$$
where $r = (N_1 / N_2)^{1/d}$ is the effective refinement ratio, $p$ is the formal order of accuracy, and $F_s = 1.25$ is the safety factor.

---

## 4. Transfer to PINN Training Datasets

Once the FEM solution is proven grid-independent:
1. **Decoupling of FEM Mesh from PINN Batching**: The PINN does not train on all $125.000$ nodes simultaneously. It sub-samples:
   - Kinematic supervision: $N_{\text{data}} \approx 2.000 - 5.000$ velocity points ($u, v$).
   - Collocation points: $N_{\text{coll}} \approx 10.000 - 20.000$ internal domain points.
2. **Autograd Sensitivity**: High-order automatic differentiation in PINNs ($\nabla^2 \mathbf{u}, \nabla \cdot \boldsymbol{\tau}$) is highly sensitive to high-frequency noise or spatial roughness in the reference data. Using a grid-independent COMSOL solution guarantees that the supervised velocity fields are infinitely smooth ($C^\infty$) and physically consistent, eliminating false PDE residual penalties.

---

## References & Back-links
- [[High_Weissenberg_Number_Problem]]
- [[ViscoelasticNet_Full model]]
- [[Viscoelastic_Training]]
- [[Sampling_Strategies]]
- [[COMSOL_Boundary_Extraction]]
- [[Owens_Phillips_Computational_Rheology]]
