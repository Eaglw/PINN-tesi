# Hazra et al. - Physics-Informed Neural Networks for Estimating Convective Heat Transfer in Jet Impingement Cooling

- **Type**: Research Paper (arXiv:2507.09356v1)
- **Date**: July 12, 2025
- **Authors**: Arijit Hazra, Prahar Sarkar, Sourav Sarkar
- **Reference**: [[Physics-Informed Neural Networks Convective Heat Transfer.pdf]]

## Summary
This paper investigates the use of PINNs for solving a complex **inverse heat transfer problem**: estimating the Convective Heat Transfer Coefficient (CHTC) at the fluid-solid interface in a jet impingement cooling setup. The authors demonstrate that PINNs can accurately infer boundary parameters from sparse and noisy temperature measurements within the solid domain, without needing to explicitly model the fluid flow.

## Key Methodology
- **Problem Formulation**: 2D transient heat conduction in a solid circular disc.
- **Nondimensionalization**: All variables (\(T, x, y, t\)) are transformed into dimensionless forms to improve PINN training stability and eliminate the need for manual data scaling.
- **PINN Architecture**: 
  - Feedforward NN with 5-6 hidden layers (40 neurons each).
  - **Activation Function**: Tanh.
  - **Library**: DeepXDE with TensorFlow backend.
- **Loss Function Components**:
  - \(L_{data}\): Sensor data MSE.
  - \(L_{PDE}\): Transient heat conduction residual.
  - \(L_{bc}\): Boundary conditions (forced/free convection, adiabatic, symmetry).
  - \(L_{init}\): Initial condition enforcement.
- **Optimization**: Adam optimizer (50,000 iterations).
- **Hyperparameter Tuning**: Optuna (Bayesian optimization) used for architecture and sampling strategy.

## Key Findings
- **Robustness to Noise**: The framework maintains high accuracy (relative error < 8%) with noise levels up to 10%. Even at 30% noise, meaningful profiles are recovered with sufficient temporal resolution.
- **Sampling Rate Impact**: Higher sampling rates significantly improve performance under high-noise conditions.
- **Implicit Regularization**: Moderate noise can sometimes improve generalization, a known effect in inverse analysis.
- **Flexibility**: The PINN can estimate both a spatially constant CHTC and a spatially varying one (parameterized as a polynomial).

## Related
- **Topics**: [[PINN_Fundamentals]], [[Inverse_Problems]], [[Nondimensionalization]]
- **Systems**: [[Heat2D_Analysis]], [[Fluid_Dynamics]]
- **Methods**: [[DeepXDE]]
