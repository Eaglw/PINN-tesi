# System: Viscoelastic Fluids

Modeling of complex fluids that exhibit both viscous and elastic characteristics.

## Governing Equations
The current implementation focuses on **Channel Flow** for an Oldroyd-B fluid:
1. **Conservation of Mass**: Automatically satisfied via [[ViscoelasticNet]].
2. **Conservation of Momentum**: 
   \[ \rho (\mathbf{u} \cdot \nabla \mathbf{u}) = -\nabla p + \mu_s \nabla^2 \mathbf{u} + \nabla \cdot \boldsymbol{\tau} \]
3. **Oldroyd-B Constitutive Equation**:
   \[ \boldsymbol{\tau} + \lambda ( \mathbf{u} \cdot \nabla \boldsymbol{\tau} - (\nabla \mathbf{u})^T \cdot \boldsymbol{\tau} - \boldsymbol{\tau} \cdot \nabla \mathbf{u} ) = \mu_p (\nabla \mathbf{u} + (\nabla \mathbf{u})^T) \]
   where \(\lambda\) is the relaxation time.

## PINN Approach (ViscoelasticNet)
As proposed in [[Thakur_et_al_ViscoelasticNet]], a multi-network architecture is used to decouple the discovery of velocity, stress, and pressure fields.
- **Model Discovery**: Treating extensibility (\(\epsilon\)) and mobility (\(\alpha\)) as trainable parameters allows the PINN to select the most appropriate constitutive model for a given dataset.
- **Backward Euler PINN**: Using temporal discretization within the loss residue to handle transient non-linear dynamics.

## Benchmark: Oldroyd-B Channel Flow
The project includes a synthetic dataset generator (`generate_dataset.py`) for stationary Poiseuille flow in a 2D channel.
- **Velocity Profile**: \(u(y) = 4 u_{max} \frac{y(H-y)}{H^2}\)
- **Pressure Profile**: Linear gradient \(\frac{dp}{dx} = \text{const}\)
- **Polymeric Stresses**:
    - \(\tau_{xy} = \mu_p \dot{\gamma}\) (Linear with shear rate)
    - \(\tau_{xx} = 2 \lambda \mu_p \dot{\gamma}^2\) (Quadratic with shear rate, capturing the "elastic" normal stress)
    - \(\tau_{yy} = 0\)
- **Purpose**: Used to validate the PINN's ability to reconstruct the stress field from sparse velocity measurements (Inverse Problem).

## Challenges
- **Numerical Instability**: High Weissenberg numbers and sharp stress gradients near corners are difficult for global networks to capture.
- **Data Sparsity**: While robust, the model requires sufficient spatio-temporal resolution (e.g., ~50,000 points) to learn complex viscosity parameters accurately.

## Related
- **Literature**: [[Thakur_et_al_ViscoelasticNet]], [[Oldroyd_B_Model]], [[Viscoelasticity_Theory]], [[Note_05_Academic_Context]]
- **Topics**: [[Viscoelasticity]], [[Fluid_Dynamics]], [[Inverse_Problems]]
