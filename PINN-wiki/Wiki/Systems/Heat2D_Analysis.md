# System: Heat2D Analysis

Research into solving the 2D Laplace equation for Heat Transfer.

## Performance Evolution
- **Baseline**: L2 error ~0.14.
- **Current Best**: L2 error < 0.007.

## Key Breakthroughs
1. **Coordinate Scaling**: Mapping domain from `[0, 1]` to **`[-1, 1]`** significantly improved gradient flow.
2. **Nondimensionalization**: Reformulating equations in dimensionless forms (as seen in [[Hazra_et_al_Convective_Heat_Transfer]]) stabilizes training by naturally scaling input/output ranges to `[0, 1]` or similar.
3. **Anchor Points**: Using ~50 supervised internal points ($\lambda_{data}=10.0$) helps break error plateaus.
3. **Boundary Density**: High density on edges ($num\_bc=400$) is essential for anchoring the solution.
4. **Funnel Architecture**: `[120, 120, 100, 80, 60, 40, 20]` provides the best compression/accuracy ratio.

## Optimization Configuration
- **Hybrid Strategy**: 2500 epochs Adam -> 1500 iterations L-BFGS.
- **Activation**: SiLU with adaptive scaling ($a=1.1$).

## References
- Chronology documented in [[Note_03_Heat2D]].
