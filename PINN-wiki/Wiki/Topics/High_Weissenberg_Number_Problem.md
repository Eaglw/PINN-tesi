# High Weissenberg Number Problem (HWNP) and Constitutive Limits

## Overview
The **High Weissenberg Number Problem (HWNP)** designates the notorious numerical instability and convergence failure that occurs in computational viscoelastic fluid dynamics (both in classical FEM/FVM solvers and Physics-Informed Neural Networks) as the Weissenberg number ($Wi = \lambda \dot{\gamma}_{\text{char}}$ or $Wi = \lambda \frac{U_{\text{ref}}}{H_{\text{ref}}}$) exceeds a moderate critical threshold ($Wi_{\text{crit}} \sim \mathcal{O}(1)$).

In complex geometries such as the **Four-Roll Mill**, stagnation points (pure extension) and tight roll gaps (intense shear) produce extreme spatial gradients in the extra-stress tensor $\boldsymbol{\tau}_p$. The severity of the HWNP and the spatial mesh resolution required to capture these fields depend fundamentally on the chosen **constitutive model**.

```
                STRESS DIVERGENCE & NUMERICAL STIFFNESS
             Oldroyd-B  >>>  Giesekus  ~  Phan-Thien-Tanner (PTT)
     ┌─────────────────────────────────────────────────────────────┐
     │  Oldroyd-B: Infinitely extensible dumbbells                │
     │             Singular extensional growth: η_E -> ∞           │
     │             Elastic boundary layer: δ ~ Wi^(-1) (Ultra-thin)│
     ├─────────────────────────────────────────────────────────────┤
     │  Giesekus:  Quadratic stress relaxation (α · τ · τ)        │
     │             Shear-thinning & bounded extensional stress     │
     │             Elastic boundary layer: δ ~ Wi^(-1/2)           │
     ├─────────────────────────────────────────────────────────────┤
     │  PTT:       Trace-dependent relaxation (1 + ε·tr(τ))        │
     │             Chain disentanglement & bounded elongation      │
     │             Elastic boundary layer: δ ~ Wi^(-1/3)           │
     └─────────────────────────────────────────────────────────────┘
```

---

## 1. Why Oldroyd-B Fails First (The Constitutive Pathology)

The Oldroyd-B model models polymer macromolecules as idealized **Hookean dumbbells** (linear entropic springs):
$$
\boldsymbol{\tau} + \lambda \overset{\triangledown}{\boldsymbol{\tau}} = 2 \eta_p \mathbf{D}
$$
where $\overset{\triangledown}{\boldsymbol{\tau}}$ is the [[Upper-convected time derivative]] and $\mathbf{D} = \frac{1}{2}(\nabla \mathbf{u} + \nabla \mathbf{u}^T)$.

### Extensional Singularity
In an extensional flow field with elongation rate $\dot{\varepsilon}$ (such as the central saddle point of the Four-Roll Mill):
$$
\mathbf{u} = (\dot{\varepsilon} x, -\dot{\varepsilon} y)
$$
The steady-state extensional stress along the stretching axis is governed by:
$$
\tau_{xx} (1 - 2 \lambda \dot{\varepsilon}) = 2 \eta_p \dot{\varepsilon} \implies \tau_{xx} = \frac{2 \eta_p \dot{\varepsilon}}{1 - 2 \lambda \dot{\varepsilon}}
$$
When the extensional strain rate approaches the critical coil-stretch transition:
$$
\dot{\varepsilon} \to \frac{1}{2\lambda} \implies \tau_{xx} \to \infty, \quad \eta_E \to \infty
$$
Because the linear dumbbell has no physical limit on how far it can stretch, the stress becomes **unbounded**. 

### Numerical Consequences
1. **Exponential Stress Layers**: Squeezing between roll walls and trailing from the central stagnation point creates exponential stress wakes that require prohibitively refined meshes.
2. **Loss of Positive Definiteness**: Discretization errors in standard Cartesian formulations easily lead to negative eigenvalues in the conformation tensor $\mathbf{A} = \mathbf{I} + \frac{\lambda}{\eta_p} \boldsymbol{\tau}$, destroying the elliptic-hyperbolic character of the equations.
3. **Severe Mesh Dependence**: Finer meshes without stabilization (DEVSS, Log-conformation) paradoxically worsen solver divergence because they resolve steeper, unphysical stress spikes.

---

## 2. Regularization in Non-Linear Models: PTT & Giesekus

To restore physical realism and numerical tractability, extended constitutive models incorporate non-linear dissipation that caps stress growth and models shear-thinning.

### The Giesekus Model (Molecular Mobility)
The Giesekus model accounts for anisotropic hydrodynamic drag and Brownian motion via a quadratic stress dissipation term:
$$
\boldsymbol{\tau} + \lambda \overset{\triangledown}{\boldsymbol{\tau}} + \frac{\alpha \lambda}{\eta_p} (\boldsymbol{\tau} \cdot \boldsymbol{\tau}) = 2 \eta_p \mathbf{D}
$$
where $\alpha \in (0, 0.5]$ is the mobility factor:
- **$\alpha = 0$**: Collapses exactly to the singular Oldroyd-B model.
- **$\alpha > 0$**: The quadratic term $\boldsymbol{\tau} \cdot \boldsymbol{\tau}$ acts as a non-linear brake. At high deformation rates, the effective polymeric viscosity experiences power-law shear-thinning ($\eta_p(\dot{\gamma}) \sim \dot{\gamma}^{-1}$), and the extensional stress asymptotes to a finite maximum plateau:
$$
\lim_{\dot{\varepsilon} \to \infty} \tau_{xx} = \frac{\eta_p}{\alpha \lambda}
$$

### The Phan-Thien–Tanner (PTT) Model (Chain Entanglement Dynamics)
Derived from network theories where polymer junctions break and reform under deformation, the linear PTT model introduces a trace-dependent relaxation coefficient:
$$
f(\operatorname{tr} \boldsymbol{\tau}) \boldsymbol{\tau} + \lambda \overset{\triangledown}{\boldsymbol{\tau}} = 2 \eta_p \mathbf{D}, \quad \text{with} \quad f(\operatorname{tr} \boldsymbol{\tau}) = 1 + \frac{\varepsilon \lambda}{\eta_p} \operatorname{tr}(\boldsymbol{\tau})
$$
where $\varepsilon \in (0, 1]$ is the extensibility parameter:
- When stresses grow, $\operatorname{tr}(\boldsymbol{\tau}) = \tau_{xx} + \tau_{yy}$ increases, automatically scaling up $f(\operatorname{tr}\boldsymbol{\tau})$.
- This dramatically accelerates the rate of stress relaxation, precluding any unbounded divergence in extensional flow and imparting shear-thinning.

---

## 3. Elastic Boundary Layer Scaling Comparison

A rigorous theoretical result in asymptotic rheology (Renardy 1997; Guy & Thomases 2015) describes the characteristic thickness $\delta_{\text{elastic}}$ of the stress boundary layers that form near solid walls and stagnation zones at high $Wi$:

| Constitutive Model | Non-linear Term | Extensional Viscosity | Boundary Layer Thickness $\delta_{\text{elastic}}$ | Tendency to Grid Convergence |
|:---|:---|:---|:---|:---|
| **Oldroyd-B / UCM** | None (Linear) | Diverges at $\dot{\varepsilon} = \frac{1}{2\lambda}$ | $\mathcal{O}(Wi^{-1})$ | **Worst** (Extremely thin layers, severe HWNP) |
| **Giesekus** | $\frac{\alpha\lambda}{\eta_p} \boldsymbol{\tau}^2$ | Bounded ($\sim 1/\alpha$) | $\mathcal{O}(Wi^{-1/2})$ | **Good to Excellent** (Robust in mixed shear/extensional flows) |
| **Linear PTT** | $\frac{\varepsilon\lambda}{\eta_p} \operatorname{tr}(\boldsymbol{\tau}) \boldsymbol{\tau}$ | Bounded ($\sim 1/\varepsilon$) | $\mathcal{O}(Wi^{-1/3})$ | **Very Good** (Smooth stress transitions, wider boundary layers) |

Because $\delta_{\text{Oldroyd-B}} \sim Wi^{-1} \ll \delta_{\text{Giesekus}} \sim Wi^{-1/2} \ll \delta_{\text{PTT}} \sim Wi^{-1/3}$, **the spatial resolution required to resolve Oldroyd-B is vastly more demanding than that required for Giesekus or PTT** at identical geometric and kinematic conditions.

---

## 4. Methodological Consequence: The "Worst-Case" Limiting Principle

In numerical experiments (both FEM in COMSOL and PINN collocation grids):
1. **Rule of Dominated Complexity**: If a spatial grid is sufficiently dense to resolve the singular, ultra-thin boundary layers of **Oldroyd-B at the maximum Weissenberg number investigated ($Wi_{\max}$)**, that identical grid is *provably guaranteed* to resolve:
   - All Oldroyd-B regimes at lower elasticity ($Wi < Wi_{\max}$).
   - All Giesekus flows ($\alpha > 0$) due to quadratic saturation and wider boundary layers ($\delta \sim Wi^{-1/2}$).
   - All PTT flows ($\varepsilon > 0$) due to trace-softening and wider boundary layers ($\delta \sim Wi^{-1/3}$).
2. **Elimination of Redundant Mesh Sweeps**: Conducting a multi-mesh convergence study (e.g. 125k, 88k, 52k, 25k) on Oldroyd-B at maximum $\lambda$ satisfies all peer-review criteria for grid independence across the entire parametric campaign.

---

## References & Back-links
- [[Mesh_Convergence_Protocol]]
- [[ViscoelasticNet_Full model]]
- [[Upper-convected time derivative]]
- [[Viscoelastic_Parameter_Identifiability]]
- [[Owens_Phillips_Computational_Rheology]]
- [[Bird_Armstrong_Hassager_Dynamics_of_Polymer_Liquids]]
- DTU Orbit: [Numerical stability of viscoelastic models](https://backend.orbit.dtu.dk/ws/portalfiles/portal/142122439/1_s2.0_S0377025717304883_main.pdf)
- Guy & Thomases (2015): [Computational Rheology & VE Fluid Modeling](https://webtool.math.ucdavis.edu/~guy/papers/guy_thomases_ve_fluid_chapt_2015.pdf)
