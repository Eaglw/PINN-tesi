# Klaudio Peqini - Solving differential equations using physically informed neural networks (PINNs)

- **Type**: Presentation (The Data Science Balkan School 2024)
- **Date**: January 22–26, 2024
- **Author**: Klaudio Peqini (University of Tirana)
- **Reference**: [[Klaudio_Peqini_PINNs.pdf]]

## Summary
This presentation provides a foundational overview of PINNs, contrasting them with traditional numerical schemes (Euler, Runge-Kutta) and standard Deep Learning interpolators. It emphasizes the "informed" nature of PINNs, where physical laws act as regularizers in the loss function.

## Key Concepts
- **Limitations of Classical Methods**: Numerical schemes like RK4 are deterministic and work well for ODEs but become computationally prohibitive for multi-scale PDEs (e.g., Navier-Stokes, MHD).
- **Interpolation vs. Extrapolation**: Standard NNs are "good interpolators but bad extrapolators." PINNs overcome this by enforcing physical constraints across the entire domain, even where data is sparse.
- **PINN Loss Structure**: The loss function is composed of domain residue and boundary/initial condition residue:
  \[ L = \omega_D L_D + \omega_B L_B \]
  where \( L_D \) is the PDE residual and \( L_B \) is the boundary/initial condition error.

## Analyzed Systems
1. **Exponential Decay**: \( y'(x) + y(x) = 0, y(0) = 1 \).
2. **Harmonic Oscillator**: \( y''(x) + y(x) = 0, y(0) = 1, y'(0) = 0 \).
3. **Korteweg-de Vries (KdV)**: Models solitons. \( y''(x) - y(x) + 3y^2(x) = 0 \).

## Conclusions
- PINNs reduce computational costs for complex differential equations.
- Modern libraries (TensorFlow, PyTorch) make implementation straightforward.
- Future application targets include Navier-Stokes and Magnetohydrodynamics (MHD).

## Related
- **Topics**: [[PINN_Fundamentals]], [[Loss_Functions]]
- **Systems**: [[Harmonic_Oscillator]], [[Fluid_Dynamics]]
