# System: Viscoelastic Fluids

Modeling of complex fluids that exhibit both viscous and elastic characteristics.

## Governing Equations
The current implementation focuses on **Channel Flow** for an Oldroyd-B fluid:
1. **Conservation of Mass**: Automatically satisfied via [[ViscoelasticNet]].
2. **Conservation of Momentum**: 
   $$ \rho (\mathbf{u} \cdot \nabla \mathbf{u}) = -\nabla p + \mu_s \nabla^2 \mathbf{u} + \nabla \cdot \boldsymbol{\tau} $$
3. **Oldroyd-B Constitutive Equation**:
   $$ \text{tau} + \text{lambda} \left( \mathbf{u} \cdot \nabla \text{tau} - (\nabla \mathbf{u}) \cdot \text{tau} - \text{tau} \cdot (\nabla \mathbf{u})^T \right) = \text{mu}_p \left( \nabla \mathbf{u} + (\nabla \mathbf{u})^T \right) $$
   where $\text{lambda}$ is the relaxation time.

### Component-wise Residuals (2D PINN)
For a 2D flow field $(u, v)$ and stress components $(\text{tau}_{xx}, \text{tau}_{xy}, \text{tau}_{yy})$, the residuals $f_{\text{tau}}$ used in the PINN loss function are derived as follows (assuming stationary state $\partial_t = 0$):

#### 1. Normal Stress $f_{\text{tau}_{xx}}$:
$$ f_{\text{tau}_{xx}} = \text{tau}_{xx} + \text{lambda} ( u \partial_x \text{tau}_{xx} + v \partial_y \text{tau}_{xx} - 2 \partial_x u \text{tau}_{xx} - 2 \partial_y u \text{tau}_{xy} ) - 2 \text{mu}_p \partial_x u $$

#### 2. Shear Stress $f_{\text{tau}_{xy}}$:
$$ f_{\text{tau}_{xy}} = \text{tau}_{xy} + \text{lambda} ( u \partial_x \text{tau}_{xy} + v \partial_y \text{tau}_{xy} - \partial_x u \text{tau}_{xy} - \partial_y u \text{tau}_{yy} - \partial_x v \text{tau}_{xx} - \partial_y v \text{tau}_{xy} ) - \text{mu}_p ( \partial_y u + \partial_x v ) $$

#### 3. Normal Stress $f_{\text{tau}_{yy}}$:
$$ f_{\text{tau}_{yy}} = \text{tau}_{yy} + \text{lambda} ( u \partial_x \text{tau}_{yy} + v \partial_y \text{tau}_{yy} - 2 \partial_x v \text{tau}_{xy} - 2 \partial_y v \text{tau}_{yy} ) - 2 \text{mu}_p \partial_y v $$

> [!IMPORTANT]
> **Bug Fix (May 2026)**: A critical bug was identified and resolved in the `tau_xy` residual implementation. Previously, the terms involving $\text{tau}_{xx}$ and $\text{tau}_{yy}$ were swapped (using $\text{tau}_{xx} \partial_y u$ instead of $\text{tau}_{yy} \partial_y u$), causing non-physical residuals for stationary Poiseuille flow where $\text{tau}_{yy}=0$ but $\text{tau}_{xx} \neq 0$.

## PINN Approach (ViscoelasticNet)
As proposed in [[Thakur_et_al_ViscoelasticNet]], a multi-network architecture is used to decouple the discovery of velocity, stress, and pressure fields.
- **Model Discovery**: Treating extensibility ($\epsilon$) and mobility ($\alpha$) as trainable parameters allows the PINN to select the most appropriate constitutive model for a given dataset.
- **Backward Euler PINN**: Using temporal discretization within the loss residue to handle transient non-linear dynamics.

## Benchmark: Oldroyd-B Channel Flow
The project includes a synthetic dataset generator (`generate_dataset.py`) for stationary Poiseuille flow in a 2D channel.
- **Velocity Profile**: $u(y) = 4 u_{max} \frac{y(H-y)}{H^2}$
- **Pressure Profile**: Linear gradient $\frac{dp}{dx} = \text{const}$
- **Polymeric Stresses**:
    - $\tau_{xy} = \mu_p \dot{\gamma}$ (Linear with shear rate)
    - $\tau_{xx} = 2 \lambda \mu_p \dot{\gamma}^2$ (Quadratic with shear rate, capturing the "elastic" normal stress)
    - $\tau_{yy} = 0$
- **Purpose**: Used to validate the PINN's ability to reconstruct the stress field from sparse velocity measurements (Inverse Problem).

## Challenges
- **Numerical Instability**: High Weissenberg numbers and sharp stress gradients near corners are difficult for global networks to capture.
- **Data Sparsity**: While robust, the model requires sufficient spatio-temporal resolution (e.g., ~50,000 points) to learn complex viscosity parameters accurately.

## Related
- **Literature**: [[Thakur_et_al_ViscoelasticNet]], [[Oldroyd_B_Model]], [[Viscoelasticity_Theory]], [[Note_05_Academic_Context]]
- **Topics**: [[Viscoelasticity]], [[Fluid_Dynamics]], [[Inverse_Problems]]
