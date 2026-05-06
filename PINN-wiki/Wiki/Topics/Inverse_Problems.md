# Topic: Inverse Problems

Inverse problems in scientific machine learning involve identifying unknown system parameters or discovering physical laws from experimental data.

## PINN Advantage
PINNs are exceptionally well-suited for inverse problems because they treat unknown parameters (e.g., thermal conductivity, reaction rates) as extra trainable variables in the neural network.

## Examples
- **Heat Transfer**: Estimating the Convective Heat Transfer Coefficient (CHTC) as seen in [[Hazra_et_al_Convective_Heat_Transfer]].
- **Fluid Dynamics**: Identifying viscosity or pressure fields from sparse velocity measurements.
- **Reaction Engineering**: Estimating Arrhenius parameters in reactor models.

## Methodology
1. Define the PDE with unknown parameter $\lambda$.
2. Augment the loss function with a data-driven term from sparse sensors.
3. Optimize weights $\theta$ and parameter $\lambda$ simultaneously.

## Related
- **Literature**: [[Hazra_et_al_Convective_Heat_Transfer]], [[Sharma_et_al_Hyperparameter_Selection]]
- **Topics**: [[PINN_Fundamentals]], [[Loss_Functions]]
