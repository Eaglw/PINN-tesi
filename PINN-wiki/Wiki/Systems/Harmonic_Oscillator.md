# Harmonic Oscillator

The Harmonic Oscillator is a fundamental physical system used to benchmark PINN performance. It is governed by a second-order linear ordinary differential equation.

## Governing Equation
The general equation for a damped harmonic oscillator is:
\[ m \frac{d^2u}{dt^2} + \mu \frac{du}{dt} + ku = 0 \]
where:
- \( m \) is the mass.
- \( \mu \) is the damping coefficient.
- \( k \) is the spring constant.

For the unit case (unit angular frequency, no damping) as discussed in [[Klaudio_Peqini_PINNs]]:
\[ y''(x) + y(x) = 0 \]
with typical initial conditions:
\[ y(0) = 1, y'(0) = 0 \]

## PINN Implementation
The PINN approach involves:
1. **Network Architecture**: A simple MLP taking time \( t \) (or position \( x \)) as input and outputting the displacement \( y \).
2. **Loss Function**:
   - **PDE Residue**: \( f = \frac{d^2\hat{y}}{dt^2} + \hat{y} \)
   - **Initial Condition Loss**: \( (\hat{y}(0) - 1)^2 + (\hat{y}'(0) - 0)^2 \)
3. **Training**: Minimizing the combined loss to find the weights that satisfy the oscillator dynamics.

## Observations
- PINNs can capture the oscillatory behavior with significantly fewer data points than standard NNs.
- Extrapolation beyond the training time window is possible but depends on the strength of the physics constraint.

## Related
- **Literature**: [[Klaudio_Peqini_PINNs]]
- **Reference**: `Reference/Harmonic oscillator PINN.ipynb`, `Reference/PINNs_maurizio.py`
- **Topics**: [[PINN_Fundamentals]]
