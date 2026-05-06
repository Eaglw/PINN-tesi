# System: Viscoelastic Fluids

Modeling of complex fluids that exhibit both viscous and elastic characteristics.

## Governing Equations
The flow of viscoelastic fluids is governed by:
1. **Conservation of Mass**: \(\nabla \cdot \mathbf{u} = 0\).
2. **Conservation of Momentum**: \(\rho (\partial_t \mathbf{u} + \mathbf{u} \cdot \nabla \mathbf{u}) = -\nabla p + \nabla \cdot \boldsymbol{\tau}'\).
3. **Constitutive Equation**: A relationship defining the extra stress tensor \(\boldsymbol{\tau}\). Common models include:
   - **Oldroyd-B**: Linear viscoelastic model.
   - **Giesekus**: Non-linear model capturing shear thinning.
   - **Linear PTT**: Captures extensional thickening.

## PINN Approach (ViscoelasticNet)
As proposed in [[Thakur_et_al_ViscoelasticNet]], a multi-network architecture is used to decouple the discovery of velocity, stress, and pressure fields.
- **Model Discovery**: Treating extensibility (\(\epsilon\)) and mobility (\(\alpha\)) as trainable parameters allows the PINN to select the most appropriate constitutive model for a given dataset.
- **Backward Euler PINN**: Using temporal discretization within the loss residue to handle transient non-linear dynamics.

## Challenges
- **Numerical Instability**: High Weissenberg numbers and sharp stress gradients near corners are difficult for global networks to capture.
- **Data Sparsity**: While robust, the model requires sufficient spatio-temporal resolution (e.g., ~50,000 points) to learn complex viscosity parameters accurately.

## Related
- **Literature**: [[Thakur_et_al_ViscoelasticNet]], [[Note_05_Academic_Context]]
- **Topics**: [[Fluid_Dynamics]], [[Inverse_Problems]]
