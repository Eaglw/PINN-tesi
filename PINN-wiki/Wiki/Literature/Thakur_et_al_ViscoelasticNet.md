# Thakur et al. - ViscoelasticNet: A physics informed neural network framework for stress discovery and model selection

- **Type**: Research Paper (arXiv:2209.06972v2)
- **Date**: June 20, 2024
- **Authors**: Sukirt Thakur, Maziar Raissi, Arezoo M. Ardekani
- **Reference**: [[ViscoelasticNet.pdf]]

## Summary
The authors present **ViscoelasticNet**, a PINN-based framework designed to discover the stress field and identify the most appropriate constitutive model for viscoelastic fluids. The framework can differentiate between non-linear models like **Oldroyd-B**, **Giesekus**, and **linear PTT** by learning extensibility (\(\epsilon\)) and mobility (\(\alpha\)) parameters from sparse velocity data.

## Key Methodology
- **Multi-Network Architecture**: Uses three specialized deep neural networks:
  - **Velocity Network (\(\phi\))**: Predicts velocity field components.
  - **Stress Network (\(\theta\))**: Predicts the components of the stress tensor.
  - **Pressure Network (\(\kappa\))**: Predicts the pressure field (solved sequentially).
- **Physics Coupling**: Combines Navier-Stokes equations with a general form of the viscoelastic constitutive equation.
- **Discretization**: Utilizes **Backward Euler** time-stepping within the PINN to construct the physics-informed residue.
- **Divergence-Free Constraint**: The velocity field is represented using a vector potential (\(\psi\)) to satisfy continuity by construction: \( u = \psi_y, v = -\psi_x \).
- **Optimization**: Multi-stage training using Adam with cosine annealing learning rate.

## Key Findings
- **Model Selection**: The framework accurately identifies the correct constitutive model based on learned physical parameters (e.g., if learned \(\epsilon, \alpha \approx 0\), the fluid is Oldroyd-B).
- **Robustness**: Works well with noisy (up to 10%) and sparse velocity data (e.g., from PIV experiments).
- **Limitations**: Struggles with sharp stress peaks at corners in complex geometries (e.g., cross-slot), suggesting a need for domain decomposition or multiple networks.

## Related
- **Topics**: [[PINN_Fundamentals]], [[Fluid_Dynamics]], [[Inverse_Problems]]
- **Systems**: [[Viscoelastic_Fluids]]
- **Methods**: [[Tapered_Architectures]]
