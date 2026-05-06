# System: Harmonic Oscillator

## Overview
The 1D Damped Harmonic Oscillator is a fundamental physics problem used to benchmark PINNs due to its well-understood analytical solution and second-order ODE nature.

The governing equation is:
\[ m \frac{d^2x}{dt^2} + \mu \frac{dx}{dt} + kx = 0 \]
where \(m\) is mass, \(\mu\) is the damping coefficient, and \(k\) is the spring constant.

## Technical Implementation
In this project, the system is used to test:
1. **Direct Discovery**: Training the PINN to match the analytical solution given initial conditions \(x(0)=1, \dot{x}(0)=0\).
2. **Inverse Parameter Identification**: Identifying \(\mu\) and \(k\) from noisy or sparse data.

### Key Refinements
- **Activation Functions**: Swish (SiLU) and GELU are preferred over Tanh to avoid gradient vanishing and capture the oscillations more accurately.
- **Regularization**: To identify parameters in the underdamped regime, a soft constraint is often added to the loss function to prevent the model from converging to an overdamped state:
  \[ \mathcal{L}_{reg} = \text{relu}(\mu - 2\sqrt{k})^2 \]
- **Optimizer Staging**: Using Adam for initialization and L-BFGS for convergence to scientific precision (\(10^{-6}\) or better).

## References
- [[Maurizio_Harmonic_Oscillator]]
- [[Klaudio_Peqini_PINNs]]
