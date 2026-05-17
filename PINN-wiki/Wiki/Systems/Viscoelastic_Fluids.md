# System: Viscoelastic Fluids

Modeling of complex fluids that exhibit both viscous and elastic characteristics.

## Governing Equations
The current implementation focuses on **Channel Flow** for an Oldroyd-B fluid:
1. **Conservation of Mass**: Automatically satisfied via [[ViscoelasticNet]].
2. **Conservation of Momentum**: 
   $$ \rho (\mathbf{u} \cdot \nabla \mathbf{u}) = -\nabla p + \mu_s \nabla^2 \mathbf{u} + \nabla \cdot \boldsymbol{\tau} $$
3. **Oldroyd-B Constitutive Equation**:
   $$ \boldsymbol{\tau} + \lambda \left( \mathbf{u} \cdot \nabla \boldsymbol{\tau} - (\nabla \mathbf{u}) \cdot \boldsymbol{\tau} - \boldsymbol{\tau} \cdot (\nabla \mathbf{u})^T \right) = \mu_p \left( \nabla \mathbf{u} + (\nabla \mathbf{u})^T \right) $$
   where $\lambda$ is the relaxation time.

### Component-wise Residuals (2D PINN)
For a 2D flow field $(u, v)$ and stress components $(\tau_{xx}, \tau_{xy}, \tau_{yy})$, the residuals $f_{\tau}$ used in the PINN loss function are derived as follows (assuming stationary state $\partial_t = 0$):

#### 1. Normal Stress $f_{\tau_{xx}}$:
$$ f_{\tau_{xx}} = \tau_{xx} + \lambda ( u \partial_x \tau_{xx} + v \partial_y \tau_{xx} - 2 \partial_x u \tau_{xx} - 2 \partial_y u \tau_{xy} ) - 2 \mu_p \partial_x u $$

#### 2. Shear Stress $f_{\tau_{xy}}$:
$$ f_{\tau_{xy}} = \tau_{xy} + \lambda ( u \partial_x \tau_{xy} + v \partial_y \tau_{xy} - \partial_x u \tau_{xy} - \partial_y u \tau_{yy} - \partial_x v \tau_{xx} - \partial_y v \tau_{xy} ) - \mu_p ( \partial_y u + \partial_x v ) $$

#### 3. Normal Stress $f_{\tau_{yy}}$:
$$ f_{\tau_{yy}} = \tau_{yy} + \lambda ( u \partial_x \tau_{yy} + v \partial_y \tau_{yy} - 2 \partial_x v \tau_{xy} - 2 \partial_y v \tau_{yy} ) - 2 \mu_p \partial_y v $$

> [!IMPORTANT]
> **Bug Fix (May 2026)**: A critical bug was identified and resolved in the `tau_xy` residual implementation. Previously, the terms involving $\text{tau}_{xx}$ and $\text{tau}_{yy}$ were swapped (using $\text{tau}_{xx} \partial_y u$ instead of $\text{tau}_{yy} \partial_y u$), causing non-physical residuals for stationary Poiseuille flow where $\text{tau}_{yy}=0$ but $\text{tau}_{xx} \neq 0$.

## PINN Approach (ViscoelasticNet)
As proposed in [[Thakur_et_al_ViscoelasticNet]], a multi-network architecture is used to decouple the discovery of velocity, stress, and pressure fields.
- **Model Discovery**: Treating extensibility ($\epsilon$) and mobility ($\alpha$) as trainable parameters allows the PINN to select the most appropriate constitutive model for a given dataset.
- **Backward Euler PINN**: Using temporal discretization within the loss residue to handle transient non-linear dynamics.

## Monitoring & Visualization
During the training process, the velocity field ($u$) and the stress components ($\tau_{xx}, \tau_{xy}, \tau_{yy}$) are plotted periodically to monitor convergence.

The **pressure field ($p$)** is omitted from periodic visualization by design. In incompressible flows, the pressure is determined by its gradient ($\nabla p$) and is typically the slowest variable to stabilize. Visualizing it in the early or intermediate stages of training provides limited physical insight until the kinematics (velocity) and constitutive (stress) fields have reached a stable state. A comprehensive validation of the pressure field is performed only at the end of the training process (post L-BFGS refinement) to ensure final physical consistency.

## Benchmark: Oldroyd-B Channel Flow
The project includes a synthetic dataset generator (`generate_dataset.py`) for stationary Poiseuille flow in a 2D channel.
- **Velocity Profile**: $u(y) = 4 u_{max} \frac{y(H-y)}{H^2}$
- **Pressure Profile**: Linear gradient $\frac{dp}{dx} = \text{const}$
- **Polymeric Stresses**:
    - $\tau_{xy} = \mu_p \dot{\gamma}$ (Linear with shear rate)
    - $\tau_{xx} = 2 \lambda \mu_p \dot{\gamma}^2$ (Quadratic with shear rate, capturing the "elastic" normal stress)
    - $\tau_{yy} = 0$
- **Purpose**: Used to validate the PINN's ability to reconstruct the stress field from sparse velocity measurements (Inverse Problem).

## Training Implementation & Debugging
For the complete technical specification of the neural network architectures, staged training orchestration, and boundary condition deduplication (geometric slicing), refer to the dedicated experiment guide: [[Viscoelastic_Training]].

## Challenges
- **Numerical Instability**: High Weissenberg numbers and sharp stress gradients near corners are difficult for global networks to capture.
- **Data Sparsity**: While robust, the model requires sufficient spatio-temporal resolution (e.g., ~50,000 points) to learn complex viscosity parameters accurately.

## Related
- **Literature**: [[Thakur_et_al_ViscoelasticNet]], [[Oldroyd_B_Model]], [[Viscoelasticity_Theory]], [[Note_05_Academic_Context]]
- **Topics**: [[Viscoelasticity]], [[Fluid_Dynamics]], [[Inverse_Problems]]
- **Systems/Experiments**: [[Viscoelastic_Training]]
