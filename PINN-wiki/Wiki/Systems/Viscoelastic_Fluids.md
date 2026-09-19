# System: Viscoelastic Fluids

Modeling of complex fluids that exhibit both viscous and elastic characteristics.

## Governing Equations
The primary production system is the **2D Four-Roll Mill** (`final_roll/`), while **1D Channel Flow** serves as a historical toy model (whose identifiability limitations are proven in [[Analisi geometria in tubo semplice]]).
1. **Conservation of Mass**: Automatically satisfied via the stream function formulation ($\nabla \cdot \mathbf{u} = 0$).
2. **Conservation of Momentum**: 
   $$ Re_{\text{scale}} (\mathbf{u} \cdot \nabla \mathbf{u}) + \nabla p = \tilde{\eta}_s \nabla^2 \mathbf{u} + \nabla \cdot \boldsymbol{\tau} $$
3. **Unified Constitutive Equations (Oldroyd-B / PTT / Giesekus)**:
   $$ f_{\text{PTT}}(\boldsymbol{\tau}) \boldsymbol{\tau} + Wi \overset{\triangledown}{\boldsymbol{\tau}} + \frac{\alpha Wi}{\tilde{\eta}_p} \boldsymbol{\tau}^2 = 2 \tilde{\eta}_p \mathbf{D} $$
   where $\lambda$ is the relaxation time, $\alpha \in [0, 0.5]$ is Giesekus mobility, and $\varepsilon \ge 0$ is PTT extensibility.

### Component-wise Residuals (2D PINN)
For a 2D flow field $(u, v)$ and stress components $(\tau_{xx}, \tau_{xy}, \tau_{yy})$, the residuals $f_{\tau}$ used in the PINN loss function are derived as follows (assuming stationary state $\partial_t = 0$):

#### 1. Normal Stress $f_{\tau_{xx}}$:
$$ f_{\tau_{xx}} = f_{\text{PTT}} \tau_{xx} + Wi ( u \partial_x \tau_{xx} + v \partial_y \tau_{xx} - 2 \partial_x u \tau_{xx} - 2 \partial_y u \tau_{xy} ) + \frac{\alpha Wi}{\tilde{\eta}_p}(\tau_{xx}^2 + \tau_{xy}^2) - 2 \tilde{\eta}_p \partial_x u $$

#### 2. Shear Stress $f_{\tau_{xy}}$:
$$ f_{\tau_{xy}} = f_{\text{PTT}} \tau_{xy} + Wi ( u \partial_x \tau_{xy} + v \partial_y \tau_{xy} - \partial_x u \tau_{xy} - \partial_y u \tau_{yy} - \partial_x v \tau_{xx} - \partial_y v \tau_{xy} ) + \frac{\alpha Wi}{\tilde{\eta}_p}\tau_{xy}(\tau_{xx} + \tau_{yy}) - \tilde{\eta}_p ( \partial_y u + \partial_x v ) $$

#### 3. Normal Stress $f_{\tau_{yy}}$:
$$ f_{\tau_{yy}} = f_{\text{PTT}} \tau_{yy} + Wi ( u \partial_x \tau_{yy} + v \partial_y \tau_{yy} - 2 \partial_x v \tau_{xy} - 2 \partial_y v \tau_{yy} ) + \frac{\alpha Wi}{\tilde{\eta}_p}(\tau_{xy}^2 + \tau_{yy}^2) - 2 \tilde{\eta}_p \partial_y v $$

## PINN Approach (ViscoelasticNet Framework)
As proposed in [[Thakur_et_al_ViscoelasticNet]] and substantially evolved in the repository (`final_roll/`):
- **Staged Inversion**: Kinematics and rheological parameters ($\lambda, \mu_p, \alpha, \varepsilon$) are discovered in Phase 1 without momentum interference; hydrodynamic pressure $p(x, y)$ is reconstructed in Phase 2 with mobile $\psi$ ([[Soft_Anti_Drift]]) and algebraic [[Pressure_Point_Anchoring]], while solvent viscosity $\mu_s$ is structurally non-identifiable from velocity/stress measurements in this geometry (see [[Solvent_Viscosity_Non_Identifiability]]).
- **Autonomous Model Discovery**: Initializing $\alpha_{\text{guess}} = 0.25$ and $\varepsilon_{\text{guess}} = 0.25$ allows the PINN to autonomously collapse or confirm non-linear constitutive terms.

## Primary Benchmark: Four-Roll Mill
The core experimental geometry is the Taylor Four-Roll Mill ($L \times H = 50\text{ mm} \times 50\text{ mm}$, roll radii $R = 5\text{ mm}$):
- **Flow Physics**: Creates a central hyperbolic stagnation point with pure extensional strain surrounded by rotational shear regions near the four counter-rotating cylinders.
- **Identifiability Advantage in Phase 1**: Unlike 1D channel flow (where $\lambda$ and $\eta_p$ decouple degenerately due to lack of longitudinal strain), the mixed shear/extensional kinematics provide an SVD condition number $\kappa \approx 1.34$, enabling full-blind recovery of constitutive parameters (see [[Viscoelastic_Parameter_Identifiability]]).
- **Solvent Viscosity Invariance**: Parametric cutline sweeps prove that varying $\eta_s$ leaves $\mathbf{u}$ and $\boldsymbol{\tau}$ completely invariant, establishing that $\eta_s$ cannot be identified without external torque or pressure measurements (see [[Solvent_Viscosity_Non_Identifiability]]).
- **Grid Independence**: High-fidelity COMSOL FEM datasets (12k, 29k, 52k, 125k nodes) validated via the [[Mesh_Convergence_Protocol]].

## Historical Note: Oldroyd-B Channel Flow
In earlier exploratory stages, stationary Poiseuille flow was evaluated. As proven in [[Analisi geometria in tubo semplice]], fully developed 1D channel flow makes the upper-convected stress derivative independent of $\lambda$, making pure velocity-driven inversion ill-posed. This led directly to the adoption of the Four-Roll Mill as the definitive scientific benchmark.

## Training Implementation & Architecture
For the complete technical specification of the neural network architectures, staged training orchestration, and boundary conditions, refer to the dedicated experiment guide: [[Viscoelastic_Training]].

## Challenges
- **Numerical Instability**: High Weissenberg numbers and sharp stress gradients near corners are difficult for global networks to capture.
- **Data Sparsity**: While robust, the model requires sufficient spatio-temporal resolution (e.g., ~50,000 points) to learn complex viscosity parameters accurately.

## Related
- **Literature**: [[Thakur_et_al_ViscoelasticNet]], [[Oldroyd_B_Model]], [[Viscoelasticity_Theory]], [[Note_05_Academic_Context]], [[Generazione_Dataset_Poiseuille]]
- **Topics**: [[Solvent_Viscosity_Non_Identifiability]], [[Viscoelasticity]], [[Fluid_Dynamics]], [[Inverse_Problems]], [[Pressure_Stress_Decoupling]]
- **Systems/Experiments**: [[Viscoelastic_Training]]
