# Topic: Loss Functions

The loss function is the central mechanism in PINN training, balancing data fidelity with physical consistency.

## Components
The total loss is typically a weighted sum:
$$ L_{total} = \omega_{data} L_{data} + \omega_{phys} L_{phys} + \omega_{bc} L_{bc} + \omega_{init} L_{init} $$

- **Data Loss ($L_{data}$)**: MSE between predictions and ground truth data.
- **Physics Loss ($L_{phys}$)**: PDE residual evaluated at collocation points.
- **Boundary Loss ($L_{bc}$)**: Enforcement of Dirichlet, Neumann, or Robin conditions.
- **Initial Loss ($L_{init}$)**: Constraint for transient problems at $t=0$.

## Weighting Strategies
The relative values of $\omega$ are crucial for convergence.
- **Dynamic Weighting**: Adjusting weights during training to balance gradient scales (see [[Dynamic_Weighting]]).
- **Integral Scaling**: Normalizing loss terms by domain volume/area to prevent one term from dominating (see [[Integral_Loss_Scaling]]).

## References
- Core structure discussed in [[PINN_Fundamentals]].
- Detailed in [[Klaudio_Peqini_PINNs]] and [[Sharma_et_al_Hyperparameter_Selection]].
