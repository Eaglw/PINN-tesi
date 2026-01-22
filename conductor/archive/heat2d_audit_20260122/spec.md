# Specification: Heat2D PINN Mathematical and Implementation Audit

## Overview
This track focuses on verifying the correctness and performance of the Physics-Informed Neural Network (PINN) implementation for the 2D Heat Transfer (Laplace) problem. The user has observed that adding the physics loss currently degrades performance, raising concerns about potential implementation errors in the equations, boundary conditions, or analytical benchmarks.

## Functional Requirements
- **Analytical Solution Verification**: Audit the `soluzione_analitica` function in `Heat2D_main.py` to ensure the Fourier series expansion correctly represents the 2D Laplace equation with the specified Dirichlet boundary conditions (T=0 on three sides, T=1 on one side).
- **PDE Residual Audit**: Verify the `heat2d_physics_loss` and `HeatEquation2D.residual` implementations, specifically checking the `torch.autograd` logic for second-order derivatives ($d^2T/dx^2 + d^2T/dy^2$).
- **Numerical Stability Check**: Evaluate the handling of `sinh` terms in the series expansion and the hard-BC ansatz to prevent numerical overflows or precision issues.
- **Performance Impact Analysis**: Investigate why the physics loss might be counterproductive, focusing on:
    - Collocation point density and distribution.
    - Loss weighting ($\lambda_{physics}$ vs $\lambda_{data}$ and $\lambda_{bc}$).
    - Training dynamics (e.g., impact of the warmup phase and L-BFGS refinement).

## Non-Functional Requirements
- **Precision**: Ensure all calculations maintain `torch.float64` for numerical stability.
- **Documentation**: Provide a detailed explanation of any corrected equations or logic.

## Acceptance Criteria
- [ ] Confirmed correctness of the analytical solution against textbook benchmarks.
- [ ] Validated PDE residual calculation via Autograd.
- [ ] Identified the root cause of performance degradation when physics loss is added.
- [ ] (If errors found) Corrected implementation of the physics loss and/or analytical solution.
- [ ] Updated results demonstrating improved or at least consistent performance with physics integration.

## Out of Scope
- Refactoring the entire project structure (limited to `Heat2D` module).
- Implementing new physics problems beyond the current 2D Heat Transfer.
