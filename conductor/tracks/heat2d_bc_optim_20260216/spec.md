# Specification: Heat2D Boundary Conditions Optimization

## Overview
This track aims to resolve training difficulties in the Heat2D problem (Laplace equation) by optimizing the sampling strategy and target values for Boundary Conditions (BC). The goal is to eliminate numerical stiffness caused by excessive point density and the approximation errors of truncated analytical series near domain corners.

## Functional Requirements
- **Refined Sampling Density**: Implement a sampling strategy that uses exactly 50 points per side (200 total boundary points).
- **Corner Singularity Mitigation**: Apply a safety margin of 0.02 from all corners ( (0,0), (Lx,0), (0,Ly), (Lx,Ly) ) to prevent the network from struggling with the 0 to 1 jump.
- **Exact Physical Targets**: Replace the `soluzione_analitica` targets for boundary points with exact constants:
    - Left side ($x=0$): $T = 0.0$
    - Right side ($x=L_x$): $T = 1.0$
    - Bottom side ($y=0$): $T = 0.0$
    - Top side ($y=L_y$): $T = 0.0$
- **Verification**: Update the overlap and point verification logic to confirm the new distribution and the absence of corner points.

## Non-Functional Requirements
- **Numerical Stability**: Reduce gradient "spikes" associated with the BC loss component.
- **Improved Convergence**: Achieve a smoother loss curve and better fit on the boundaries by removing the "noise" of the truncated Fourier series.

## Acceptance Criteria
- [ ] Boundary points count is exactly 200 (50 per side).
- [ ] No boundary point is sampled within a distance of 0.02 from any corner.
- [ ] BC loss target values are purely 0.0 or 1.0.
- [ ] The model shows improved accuracy on the boundary regions in the final error maps compared to the 100-points baseline.

## Out of Scope
- Modification of the internal collocation points (Physics points).
- Changes to the core NN architecture (layers, activation functions) beyond testing the impact of this optimization.
- Implementation of "Hard" Boundary Conditions (architectural constraints).
