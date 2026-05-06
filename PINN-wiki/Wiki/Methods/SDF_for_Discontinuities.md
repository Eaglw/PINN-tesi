# Method: SDF for Discontinuities

The Signed Distance Function (SDF) is a technique used to handle discontinuities in boundary conditions within the PINN framework.

## Problem Statement
Standard PINNs struggle with sharp transitions in BCs (e.g., a jump in temperature at a corner) because neural networks are inherently continuous functions.

## Implementation
An SDF is used to weight the loss points near the region of discontinuity. Points very close to the jump are assigned lower weights, preventing the optimizer from being overwhelmed by the infinite gradient at the point of discontinuity.

## References
- Key technique for stiff heat conduction problems discussed in [[Sharma_et_al_Hyperparameter_Selection]].
