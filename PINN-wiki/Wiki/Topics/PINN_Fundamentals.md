# PINN Fundamentals

Physics-Informed Neural Networks (PINNs) are a class of deep learning models designed to solve differential equations by embedding physical laws directly into the neural network's loss function.

## Core Theory
Standard neural networks act as universal function approximators, but they often require large amounts of data and can fail to generalize outside the training range (bad extrapolators). PINNs address this by using the governing physical equations (ODEs or PDEs) as a form of regularization.

As discussed in [[Klaudio_Peqini_PINNs]], PINNs leverage the "informed" nature of physics to identify patterns that are not merely visual or statistical, but are grounded in the underlying dynamics of the system.

## The PINN Loss Function
The total loss in a PINN training process typically combines data-driven loss with physics-based residuals:
$$ L_{total} = \omega_{data} L_{data} + \omega_{phys} L_{phys} + \omega_{bc} L_{bc} $$

- **Data Loss ($L_{data}$)**: Mean Squared Error (MSE) between the network prediction and available experimental or simulation data.
- **Physics Loss ($L_{phys}$)**: The residual of the differential equation evaluated at collocation points (often sampled via [[Sampling_Strategies]]).
- **Boundary/Initial Condition Loss ($L_{bc}$)**: Ensures the solution satisfies the specific boundary and initial conditions of the problem.

## Key Methodologies

### Inverse Problems
One of the most powerful applications of PINNs is solving inverse problems, where unknown parameters in the physical equation (e.g., CHTC in [[Hazra_et_al_Convective_Heat_Transfer]]) are treated as trainable variables. By minimizing the discrepancy with sparse sensor data ($L_{data}$), PINNs can infer these parameters while ensuring the overall field satisfies physical laws.

### Nondimensionalization
Representing physical equations in nondimensional form is a common preprocessing step in PINNs. It scales the variables to a similar order of magnitude (typically `[0, 1]` or `[-1, 1]`), which:
- Improves numerical stability.
- Speeds up convergence.
- Reduces the need for manual weight tuning for different loss terms.
- Ensures consistent scaling between data-driven and physics-based losses.

## Advantages over Classical Methods
1. **Mesh-free**: PINNs do not require a traditional computational grid, making them suitable for complex geometries.
2. **Inverse Problems**: PINNs are naturally suited for parameter identification (finding unknown coefficients in the PDE).
3. **Data Efficiency**: Physical constraints significantly reduce the amount of labeled data required for accurate training.

## Limitations
- **Hyperparameter Sensitivity**: The weights ($\omega$) for different loss terms are critical and often require tuning (see [[Dynamic_Weighting]]).
- **Convergence**: Training can be slower than traditional methods for well-behaved problems, often requiring multi-stage optimization (see [[Staged_Precision_Strategy]]).

## Related
- **Literature**: [[Klaudio_Peqini_PINNs]], [[Note_01_Framework]]
- **Methods**: [[Dynamic_Weighting]], [[Staged_Precision_Strategy]]
