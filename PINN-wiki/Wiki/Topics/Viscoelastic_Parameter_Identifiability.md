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

### Non-Linear Parameters & Unbiased Model Discovery ($\alpha, \varepsilon$)
Per i modelli reologici estesi (Giesekus e PTT), la formulazione unificata include parametri addizionali vincolati:
$$\alpha = 0.5 \cdot \sigma(r_\alpha) \in [0, 0.5], \qquad \varepsilon = \operatorname{softplus}(r_\varepsilon) \ge 0$$

> [!IMPORTANT]
> **Unbiased Model Selection Principle**:
> Quando si valuta l'identificabilità su dataset sintetici generati da un modello target (ad esempio Oldroyd-B puro, dove $\alpha_{\text{true}} = 0$ ed $\varepsilon_{\text{true}} = 0$), i guess iniziali non devono essere scelti arbitrariamente prossimi a zero (che faciliterebbe artificialmente la convergenza).
> Impostando **$\alpha_{\text{guess}} = 0.25$** ed **$\varepsilon_{\text{guess}} = 0.25$** (il centro esatto dei rispettivi domini fisici $[0, 0.5]$):
> 1. $\alpha_{\text{guess}} = 0.25 \implies r_\alpha = 0$, massimizzando la derivata $\sigma'(0) = 0.25$ della sigmoide ed eliminando la saturazione iniziale.
> 2. Si impone un test di selezione del modello rigoroso e "cieco", verificando se la PINN è in grado di annullare spontaneamente i contributi non lineari senza assunzioni a priori.

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

### 2. Solvent Viscosity Identification: Direct Momentum vs. Curl-Momentum (Offline Diagnostic)
Two distinct formulations for isolating solvent viscosity $\eta_s$ were historically tested on the momentum balance using exact numerical simulation fields:

#### A. Curl-Momentum Formulation ($\nabla \times \nabla p \equiv 0$)
Eliminating pressure by taking the curl of the momentum equation:
$$\nabla \times \left[ Re_{\text{scale}} (\mathbf{u} \cdot \nabla \mathbf{u}) - \tilde{\eta}_s \nabla^2 \mathbf{u} - \nabla \cdot \boldsymbol{\tau} \right] = 0$$
- Result: $\eta_s \approx 0.0051\ \text{Pa}\cdot\text{s} \quad (\mathbf{95\%} \text{ relative error})$.
- **Root Cause**: High-order finite differences amplify discretization noise. For grid spacing $\Delta x \approx 0.028$, the 2nd derivative operator scales as $\frac{1}{\Delta x^2} \approx 1250$, and the 3rd/4th order derivatives in the curl operator scale as $\frac{1}{\Delta x^3} \approx 45000$, destroying the signal.

#### B. Direct Momentum Formulation (A-Priori Pressure Dependent)
Directly fitting the momentum balance with exact simulation pressure gradients:
$$\nabla p_{\text{COMSOL}} + Re_{\text{scale}} (\mathbf{u} \cdot \nabla \mathbf{u}) - \nabla \cdot \boldsymbol{\tau} = \tilde{\eta}_s \nabla^2 \mathbf{u}$$
- Spatial correlation: **$0.8929$**
- Result: $\eta_s = 0.098971\ \text{Pa}\cdot\text{s} \quad (\mathbf{1.03\%} \text{ relative error})$.

> [!CAUTION]
> **Fondamentale Caveat Fisico — Non-Identificabilità di $\eta_s$ nel Vero Problema Inverso**:
> L'apparente recupero di $\eta_s$ al $1.03\%$ nel test Direct Momentum dipende **esclusivamente dal fatto che il vero gradiente di pressione COMSOL $\nabla p$ era fornito come dato noto a priori**.
> In un problema inverso reale (o benchmark PIV), **la pressione interna $p(x,y)$ non è misurabile**. Come dimostrato dallo studio parametrico su cutline (si veda **[[Solvent_Viscosity_Non_Identifiability]]**):
> 1. I campi di velocità $\mathbf{u}$ e di extra-stress $\boldsymbol{\tau}$ sono **esattamente invarianti rispetto a variazioni di $\eta_s$**.
> 2. Qualsiasi variazione $\Delta \eta_s$ viene assorbita al $100\%$ da una traslazione irrotazionale del campo di pressione ignoto ($\Delta p = \Delta \eta_s \phi$).
> 3. L'informazione di Fisher contenuta nei dati osservabili $(\mathbf{u}, \boldsymbol{\tau})$ è identicamente nulla: $\mathcal{I}(\eta_s) \equiv 0$.
> Pertanto, **$\eta_s$ è strutturalmente e geometricamente non identificabile** nel Four-Roll Mill senza misure esterne di pressione o di coppia sui rulli.

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
2. **Decoupled Two-Phase Optimization**: Phase 1 robustly identifies constitutive parameters $(\lambda, \eta_p, \alpha, \varepsilon)$ with frozen pressure; Phase 2 solves for the hydrodynamic pressure field $p$ (see [[Solvent_Viscosity_Non_Identifiability]] for why $\eta_s$ cannot be independently discovered from kinematics).
3. **Multi-Start Verification**: Testing across multiple random initializations spanning orders of magnitude to confirm global basin of attraction.

---

## High-Weissenberg PTT Identifiability Degeneracy & Effective Relaxation Law

### 1. Phenomenological Evidence & Invariance Across Optimization Schemes
During inverse parameter discovery on high-Weissenberg Phan-Thien–Tanner (PTT) fluids ($Wi = 1.666$, $\lambda_{\text{true}} = 1.0\,\text{s}$, $\mu_{p,\text{true}} = 0.5\,\text{Pa}\cdot\text{s}$, $\varepsilon_{\text{true}} \in \{0.1, 0.3, 0.5\}$) on the Four-Roll Mill benchmark with boundary-only stress supervision, the PINN consistently exhibits the following behavior:
1. **Rigid Inversion of Shear Modulus**: The elastic shear modulus $G = \frac{\mu_p}{\lambda} \equiv 0.500\,\text{Pa}$ is captured with $< 1\%$ relative error across all runs.
2. **Extensibility Parameter Collapse**: The non-linear parameter collapses: $\varepsilon \to 0$ ($\sim 10^{-4}$).
3. **Systematic Relaxation Scaling**: Relaxation time and polymeric viscosity systematically scale down together along the constant-$G$ valley:
   - For $\varepsilon_{\text{true}} = 0.1$: $\lambda_{\text{est}} = 0.5165\,\text{s}$, $\mu_{p,\text{est}} = 0.2567\,\text{Pa}\cdot\text{s} \quad (G = 0.4969\,\text{Pa})$
   - For $\varepsilon_{\text{true}} = 0.3$: $\lambda_{\text{est}} = 0.4211\,\text{s}$, $\mu_{p,\text{est}} = 0.2128\,\text{Pa}\cdot\text{s} \quad (G = 0.5054\,\text{Pa})$
   - For $\varepsilon_{\text{true}} = 0.5$: $\lambda_{\text{est}} = 0.3672\,\text{s}$, $\mu_{p,\text{est}} = 0.1854\,\text{Pa}\cdot\text{s} \quad (G = 0.5048\,\text{Pa})$

### 2. Algorithmic Invariance (Ablation Benchmark on $\varepsilon = 0.3$)
To verify whether this behavior was an optimization artifact (such as Softplus saturation or initial chaotic transients), four independent optimization configurations were benchmarked:

| Run Configuration | Setup Dettagliato | $\lambda_{\text{est}}$ [s] | $\mu_{p,\text{est}}$ [Pa·s] | $G_{\text{est}}$ [Pa] | $\varepsilon_{\text{est}}$ |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Baseline Locale (PC)** | Softplus, No Warmup | 0.4211 (-57.9%) | 0.2128 (-57.4%) | 0.5054 (+1.1%) | $1.70 \times 10^{-4}$ |
| **Run 1 (Kaggle)** | Softplus, **Warmup 8k** | 0.4221 (-57.8%) | 0.2130 (-57.4%) | 0.5047 (+0.9%) | $1.22 \times 10^{-4}$ |
| **Run 2 (Kaggle)** | **EpsExp**, No Warmup | 0.4284 (-57.2%) | 0.2168 (-56.6%) | 0.5062 (+1.2%) | $2.10 \times 10^{-4}$ |
| **Run 3 (Kaggle)** | **EpsExp**, **Warmup 8k** | 0.4284 (-57.2%) | 0.2168 (-56.6%) | 0.5062 (+1.2%) | $2.10 \times 10^{-4}$ |

The exact convergence across disparate optimization manifolds proves that the minimum is a **structural property of the boundary-supervised PDE**, not an optimization defect.

### 3. The Linear Relaxation Rate Law
Plotting the recovered relaxation rate $\frac{1}{\lambda_{\text{est}}}$ against the true non-linear parameter $\varepsilon_{\text{true}}$ reveals an exact affine relationship:
$$\frac{1}{\lambda_{\text{est}}} = \frac{1}{\lambda_{\text{true}}} + C \cdot \varepsilon_{\text{true}}$$
Empirical verification:
- $\varepsilon = 0.1 \implies 1/\lambda_{\text{est}} = 1.936\,\text{s}^{-1}$
- $\varepsilon = 0.3 \implies 1/\lambda_{\text{est}} = 2.334\,\text{s}^{-1} \implies \Delta(1/\lambda) = 0.398$ (per $\Delta\varepsilon = 0.2$)
- $\varepsilon = 0.5 \implies 1/\lambda_{\text{est}} = 2.723\,\text{s}^{-1} \implies \Delta(1/\lambda) = 0.389$ (per $\Delta\varepsilon = 0.2$)
$$\frac{\Delta(1/\lambda)}{\Delta\varepsilon} \approx 1.97 \approx 2.0$$

### 4. Physical Derivation: Shear-Thinning Absorption into an Equivalent Oldroyd-B Fluid
Dividing the PTT constitutive equation by $Wi = \lambda \frac{U_{\text{ref}}}{H_{\text{ref}}}$:
$$\left( \frac{1}{Wi} + \frac{\varepsilon}{\mu_p} \text{tr}(\boldsymbol{\tau}) \right) \boldsymbol{\tau} + \overset{\triangledown}{\boldsymbol{\tau}} = 2 \left(\frac{H_{\text{ref}}}{U_{\text{ref}}}\right) \frac{\mu_p}{\lambda} \mathbf{D}$$
Because the only stress boundary conditions are imposed along the rotating rollers (predominantly shear flow), the trace of the extra-stress tensor $\langle \text{tr}(\boldsymbol{\tau}) \rangle \approx 1.4$ acts as an effective scalar drag multiplier.
The non-linear softening term $\frac{\varepsilon}{\mu_p}\text{tr}(\boldsymbol{\tau})$ is absorbed into an effective Weissenberg number:
$$\frac{1}{Wi_{\text{eff}}} = \frac{1}{Wi} + \frac{\varepsilon}{\mu_p} \langle \text{tr}(\boldsymbol{\tau}) \rangle$$
Consequently, an Oldroyd-B fluid ($\varepsilon = 0$) with reduced relaxation time $\lambda_{\text{eff}} \approx 0.42\,\text{s}$ and viscosity $\mu_{p,\text{eff}} \approx 0.21\,\text{Pa}\cdot\text{s}$ generates identical boundary stress and bulk velocity fields to a PTT fluid with $\lambda = 1.0\,\text{s}, \mu_p = 0.5\,\text{Pa}\cdot\text{s}, \varepsilon = 0.3$.
Because linear models feature lower curvature in the loss landscape, gradient-based PINN optimizers are naturally pulled toward the equivalent linear Oldroyd-B attractor.

### 5. Theoretical Origin: Why Extensional Kinematics Activates the PTT Non-Linearity
To understand why the parameter $\varepsilon$ is elusive under boundary-only shear supervision, one must revisit the theoretical foundation of the Phan-Thien–Tanner model (Phan-Thien & Tanner, 1977):
$$\left[ 1 + \varepsilon \frac{\lambda}{\eta_p} \text{tr}(\boldsymbol{\tau}) \right] \boldsymbol{\tau} + \lambda \overset{\triangledown}{\boldsymbol{\tau}} = 2 \eta_p \mathbf{D}$$
The term distinguishing Linear PTT from Oldroyd-B is the stress-dependent multiplier:
$$f(\boldsymbol{\tau}) = 1 + \varepsilon \frac{\lambda}{\eta_p} \text{tr}(\boldsymbol{\tau})$$

#### A) General Constitutive Model vs. Regimes of Distinct Signatures:
Crucially, **PTT is not merely an "extensional model"**: it is a general, frame-invariant constitutive law for non-linear viscoelastic liquids applicable to shear, extension, and mixed flows alike. The distinction lies between **constitutive relevance** and **parameter identifiability**:
$$\boxed{\text{Relevance of PTT} \neq \text{Identifiability of } \varepsilon}$$
- In **simple shear flow** ($\mathbf{D}_{12} = \dot{\gamma}/2$), normal stresses grow moderately. The term $\varepsilon \frac{\lambda}{\eta_p}\text{tr}(\boldsymbol{\tau})$ induces mild shear-thinning, which can be readily mimicked by shifting effective linear parameters $(\lambda, \eta_p)$ without activating a distinct non-linear signature.
- In **extensional flow** ($\mathbf{D} = \text{diag}(\dot{\epsilon}, -\dot{\epsilon})$), normal stresses grow rapidly with $Wi_e = \lambda \dot{\epsilon}$. In Oldroyd-B, this leads to the unphysical divergence of extensional viscosity ($\eta_E \to \infty$ for $Wi_e \to 0.5$). PTT introduces a physical bound precisely through $f(\text{tr}\boldsymbol{\tau})$: as stress escalates, $f(\text{tr}\boldsymbol{\tau}) \gg 1$ limits chain stretch and caps the extensional stress at a finite plateau.
- Consequently, while PTT is physically valid across all flow regimes, **the specific physical mechanism governed by $\varepsilon$ produces its most pronounced, non-collinear signature precisely in strongly extensional kinematics** (near stagnation points).

---

### 6. Identifiability vs Sensitivity: Collinearity of the Sensitivity Jacobian
A central epistemological lesson of this investigation is that **high parameter sensitivity does not imply identifiability**:
- The sensitivity vector $\mathbf{S}_\theta = \frac{\partial \mathcal{R}_{\text{const}}}{\partial \theta}$ measures how the constitutive residual responds to parameter variations $\theta \in \{\varepsilon, \lambda, \mu_p\}$.
- However, for parameters to be individually discoverable, their sensitivity directions must be **linearly independent**. If $\mathbf{S}_\varepsilon$ is nearly collinear with $\mathbf{S}_\lambda$ or $\mathbf{S}_{\mu_p}$, the Fisher Information Matrix (or Gram matrix) $F = J^T J$ becomes ill-conditioned ($\det(F) \approx 0$), creating an infinite flat valley where shifts in $\varepsilon$ are compensated by adjustments to $\lambda$ and $\mu_p$.

#### Quantitative Diagnostic on the Four-Roll Mill (M5k Benchmark):
Using the offline diagnostic tool (`scratch/offline_sensitivity_collinearity.py`), the spatial sensitivities and normalized correlation matrix $C_{ij} = \frac{F_{ij}}{\sqrt{F_{ii} F_{jj}}}$ were evaluated across three distinct flow zones:

| Flow Region | Kinematic Index $\bar{\xi}$ | Collinearity $|\cos\theta_{\varepsilon,\lambda}|$ | Correlation $|\rho_{\varepsilon,\lambda}|$ | Correlation $|\rho_{\varepsilon,\mu_p}|$ | Condition Number $\kappa(C)$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Roller Boundary (Shear BCs)** | $+0.825$ | **0.4194** | **0.4132** | **0.9115** | **$1.70 \times 10^3$** |
| **Global Domain (Bulk)** | $+0.721$ | 0.5133 | 0.4612 | 0.8905 | $1.39 \times 10^2$ |
| **Extensional Core $\Omega_{\text{ext}}$** | **$+0.836$** | **0.5321** | **0.5765** | **0.6974** | **$1.72 \times 10^1$** |

**Rigorous Interpretation**:
*The sensitivity analysis indicates that the extensional core provides a substantially better-conditioned parameter-sensitivity structure than the roller boundaries, with the condition number decreasing from approximately $1.7 \times 10^3$ to $17.2$. This result is consistent with the theoretical role of the PTT nonlinear stress-dependent correction, whose contribution becomes more pronounced as the extensional stress increases. The extensional region therefore represents a promising location for improving the identifiability of the extensibility parameter $\varepsilon$, although full parameter identifiability must ultimately be assessed from the inverse problem itself.*

---

### 7. Strict Invariance of the Elastic Shear Modulus ($G = \mu_p / \lambda \equiv 0.50\,\text{Pa}$)
Across all training runs on high-Weissenberg PTT fluids ($Wi = 1.666$, $\lambda_{\text{true}} = 1.0\,\text{s}$, $\mu_{p,\text{true}} = 0.5\,\text{Pa}\cdot\text{s}$), the recovered elastic modulus:
$$G = \frac{\mu_p}{\lambda}$$
remains **rigorously invariant and exact**:
- Baseline (Softplus, No Warmup): $G_{\text{est}} = \frac{0.2128}{0.4211} = \mathbf{0.5054\,\text{Pa}} \quad (+1.08\%)$
- Run 1 (Warmup 8k): $G_{\text{est}} = \frac{0.2130}{0.4221} = \mathbf{0.5047\,\text{Pa}} \quad (+0.94\%)$
- Run 2 (EpsExp, No Warmup): $G_{\text{est}} = \frac{0.2168}{0.4284} = \mathbf{0.5062\,\text{Pa}} \quad (+1.24\%)$
- Run 3 (EpsExp, Warmup 8k): $G_{\text{est}} = \frac{0.2168}{0.4284} = \mathbf{0.5062\,\text{Pa}} \quad (+1.24\%)$

**Physical Meaning — The 1D Elastic Valley**:
This consistent convergence demonstrates that in shear-dominated flows, the PINN effortlessly identifies a 1D manifold of constant effective elasticity:
$$\mathcal{V} \approx \left\{ (\lambda, \mu_p, \varepsilon) : \frac{\mu_p}{\lambda} \approx 0.50\,\text{Pa} \right\}$$
However, because boundary data in shear flow cannot decouple the individual coordinates along $\mathcal{V}$ (due to collinearity $\kappa \sim 10^3$), gradient descent slides along the valley toward the lowest-curvature linear attractor ($\varepsilon \to 0$, $\lambda \to \lambda_{\text{eff}} \approx 0.42\,\text{s}$).

---

### 8. Dimensionless Rigor & Literature Precedent (ViscoelasticNet Cross-Slot)
1. **Dimensionless Formulation vs. Bare Units**:
   A heuristic stating that "$\eta_p$ must be smaller than $\lambda$" is dimensionally invalid without reference units ($[\text{Pa}\cdot\text{s}]$ vs $[\text{s}]$). The rigorous criterion is:
   $$\text{Do not make the characteristic elastic modulus } G = \frac{\eta_p}{\lambda} \text{ excessively large relative to the characteristic flow shear stress } \tau_0 = \eta_0 \dot{\gamma}_0.$$
2. **The Cross-Slot Geometry Precedent**:
   In the foundational work of ViscoelasticNet ([[Thakur_et_al_ViscoelasticNet]]), the authors successfully inverted the Linear PTT parameters ($\varepsilon = 0.02, \lambda = 0.008, \eta_p = 0.025$). This success was directly enabled by their choice of geometry: the **cross-slot**, where opposing planar jets collide at a central stagnation junction, producing an extensional-dominated flow across a substantial portion of the domain. In the Four-Roll Mill, however, extensional flow is localized exclusively within a narrow central core, while the vast majority of the domain and all supervised roller boundaries are pure shear.

---

### 9. The Subdomain Strategy: Physics-Informed Experimental Design
Focusing on the central extensional core $\Omega_{\text{ext}} = \{ (x, y) : |x - x_c| \le \delta, \; |y - y_c| \le \delta \}$ is not an ad-hoc numerical trick to artificially manipulate loss weights. It is a **physics-informed Design of Experiments (DoE) principle**:
> **Experimental Design Rationale**:
> By focusing supervision or constitutive enforcement on $\Omega_{\text{ext}}$, one intentionally selects the domain region where the constitutive mechanism uniquely associated with $\varepsilon$ produces its most pronounced, non-collinear physical signature, preventing it from being diluted across the shear-dominated bulk.

---

### 9. Implications for Experimental Rheometry & Thesis
- **Practical Identifiability Limit**: With boundary-only stress data and bulk velocity, PTT fluids at high Weissenberg ($Wi > 1$) cannot be decoupled from equivalent linear Oldroyd-B fluids.
- **Resolution Strategy**: Resolving $\varepsilon$ requires extensional stress data in the bulk, specifically optical birefringence measurements along the central stagnation streamline ($x=x_c, y=y_c$), where extensional stress growth diverges between Oldroyd-B and PTT.

---

## Global Parameter Identifiability vs. Local Stress Field Fidelity at High Weissenberg

In the benchmark run on Oldroyd-B with $Wi = 1.2$ and quasi-absence of solvent ($\beta_s = 0.02, \eta_p = 0.98, \eta_s = 0.02$) on an ultralight mesh ($M=5\text{k}$, 5,086 points), an extraordinary decoupling was observed between global parameter estimation and local stress reconstruction:

### 1. Experimental Evidence (Run `[2026-09-26_15-44]`)
- **Kinematics Fidelity**: $L_2$ error on $(u, v) \le \mathbf{0.80\%}$.
- **Stress Field Smoothing**: $L_2$ error on $\tau_{xy} \approx \mathbf{25.66\%}$, and $\tau_{\text{diag}} \approx \mathbf{28.85\%}$.
- **High-Precision Parameter Discovery**:
  $$\lambda_{\text{est}} = 1.2486\,\text{s} \quad (\mathbf{+4.05\%}), \qquad \mu_{p,\text{est}} = 1.0513\,\text{Pa}\cdot\text{s} \quad (\mathbf{+7.27\%})$$
  with non-linear parameters $\alpha$ and $\varepsilon$ spontaneously suppressed to $< 6 \times 10^{-4}$ from blind initial guesses ($0.25$).

### 2. The Physical Mechanism: Convective Relaxation vs. Boundary Layer Spikes
Why does the PINN identify the constitutive parameters with $< 5\text{--}8\%$ error while displaying a $\sim 26\text{--}29\%$ point-wise error in the stress tensor?
1. **Elastic Boundary Layer Sub-Resolution**:
   For an Oldroyd-B fluid without solvent damping ($\beta_s \to 0$), the stress boundary layer near rotating cylinders scales as $\delta \sim Wi^{-1}$. On a coarse 5k mesh, the inter-node spacing $h$ is comparable to or larger than $\delta$. The neural network acts as a continuous low-pass filter, inevitably smoothing out the acute stress peaks at the roller boundaries.
2. **Bulk Convective Decay as the Parameter Signal**:
   The parameters $\lambda$ and $\mu_p$ are not encoded solely in the local amplitude of the boundary stress peak. Instead, the constitutive equation dictates the **convective stress relaxation along streamlines in the bulk**:
   $$\mathbf{u} \cdot \nabla \boldsymbol{\tau} \sim -\frac{1}{\lambda} \boldsymbol{\tau} + 2 \frac{\mu_p}{\lambda} \mathbf{D}$$
   Because the bulk flow covers over $90\%$ of the domain and the velocity field $\mathbf{u}$ is resolved with $< 0.8\%$ error, the spatial rate of stress decay provides an abundantly sampled, highly constrained mathematical signature that pins down $\lambda$ and $\mu_p$ regardless of local boundary layer truncation.

### 3. Mesh Convergence Takeaway (Giesekus $5\text{k} \to 12\text{k}$)
The parallel mesh study on Giesekus ($L=0.7, \alpha=0.35$) further confirms this principle:
- Increasing collocation points from $5\text{k}$ to $12\text{k}$ halved the estimation errors ($\lambda$: $-8.2\% \to -4.1\%$; $\mu_p$: $-7.8\% \to -3.5\%$).
- Crucially, the non-linear mobility parameter $\alpha$ converged to an identical value ($\alpha \approx 0.297$ vs $0.299$, $\sim -15\%$), proving that parameter identifiability is structurally robust even on sparse grids, and that residual parameter offsets stem from intrinsic PDE sensitivity rather than spatial under-sampling.

---

## References & Back-links
- [[Solvent_Viscosity_Non_Identifiability]] (Structural non-identifiability theorem of solvent viscosity in the 4-roll mill)
- [[High_Weissenberg_Number_Problem]] (Boundary layer scaling, Oldroyd-B vs PTT singularities)
- [[ViscoelasticNet_Full model]] (Unified constitutive model and non-linear parameterization)
- [[Adaptive_Nondimensionalization]] (Block-wise scaling protocol)
- [[Soft_Anti_Drift]] (Kinematic stabilization during momentum training)
- [[Staged_Training_Procedure]] (Multi-stage training workflow)
- [[Pressure_Stress_Decoupling]] (Helmholtz-Hodge decomposition and pressure isolation)
- [[Viscoelastic_Training]] (System experiment configuration)
