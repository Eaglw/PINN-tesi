# Note: Heat2D Path to Precision

- **Type**: Technical Journal
- **Subject**: 2D Laplace Equation (Heat Transfer) Optimization
- **Reference**: `Reference/Note_mio_studio/03_Heat2D_Research_Journal.md`

## Summary
This document traces the research evolution of the Heat2D problem, documenting the transition from baseline errors (~0.14) to high-precision results (L2 < 0.007) through automated search and architectural refinements.

## Key Breakthroughs
- **Funnel Architecture**: Identifying `[120, 120, 100, 80, 60, 40, 20]` as the optimal configuration for Laplacian problems.
- **Coordinate Scaling**: Mapping the domain to **`[-1, 1]`** had the most significant impact on gradient flow and accuracy.
- **Anchor Points**: Using 50 supervised internal points with high weight ($\lambda=10.0$) acts as a "compass" to guide the optimizer out of local minima.
- **Sampling**: Moving to **Pure Sobol sampling** (1600 points) and high boundary density (400 points) was foundational for precision.

## Optimization Configuration
- **Hybrid Strategy**: 2500 Adam epochs -> 1500 L-BFGS iterations.
- **Activation**: **SiLU** with adaptive scaling ($a=1.1$).

## Related
- **Systems**: [[Heat2D_Analysis]]
- **Topics**: [[Sampling_Strategies]], [[Activation_Functions]], [[PINN_Fundamentals]]
- **Methods**: [[Tapered_Architectures]], [[Staged_Precision_Strategy]]
