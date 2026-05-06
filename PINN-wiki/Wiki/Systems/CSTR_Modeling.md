# System: CSTR Modeling

Modeling of a Continuous Stirred-Tank Reactor (CSTR) using PINNs.

## Approaches
- **Direct Problem**: Predicting concentration and temperature over time.
- **Inverse Problem**: Estimating reaction parameters (e.g., reaction rate constants).

## Coupled PINN (Non-Isothermal)
Used for systems where concentration and temperature are interdependent.
- **Architecture**: Dual networks (`ConcentrationNet` and `TemperatureNet`).
- **Coupling**: The Arrhenius term \( k(T) = k_0 e^{-E/RT} \) links the mass and energy balances.

## Training Strategies
- **Warm-up**: Setting \(\lambda_{phys} = 0\) for the first ~1000 steps to stabilize thermal ranges and prevent Arrhenius term explosion.
- **Hybrid Optimization**: Adam for initial approach, L-BFGS for refinement.

## References
- Detailed in [[Note_02_CSTR]].
