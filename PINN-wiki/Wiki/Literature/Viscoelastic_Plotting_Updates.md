# Viscoelastic Plotting Updates

## Summary
Technical walkthrough documenting the overhaul of the visualization and logging pipeline for Viscoelastic Oldroyd-B simulations. The focus is on improving the interpretability of multi-field results and robustness of error visualization across different training regimes (PurePhys, Phys+Data).

## Key Methodology
- **Multi-field Grid Visualization**: Implementation of `plot2D_viscoelastic_final()` which generates a 5x3 grid comparing Predictions, Exact solutions, and Relative Error maps for velocity ($u$), pressure ($p$), and the three stress components ($\tau_{xx}, \tau_{xy}, \tau_{yy}$).
- **Adaptive Error Scaling**: Replacement of hardcoded `vmax` values with an adaptive strategy based on the 95th percentile of the error distribution: `vmax = max(np.percentile(95), 1.0)`. This prevents "saturation" (entirely red maps) when errors are high.
- **Cross-Model Comparison**: `plot2D_viscoelastic_comparison()` enables side-by-side error map evaluation across different training goals (e.g., PurePhys vs. DataPhys).
- **Metric Aggregation**: Enhanced logging in `results.csv` using `L2_avg` (mean of relative errors across all fields) and `Max_global` (the maximum error recorded across the entire domain and all fields).

## Key Findings
- **Visual Clarity**: Differentiating colormaps (`inferno` for velocity, `viridis` for pressure, `plasma` for stress) significantly improves the ability to distinguish physical fields in complex plots.
- **Error Sensitivity**: Adaptive $V_{max}$ is critical for debugging early-stage models or stiff problems where errors might exceed standard fixed ranges.
- **Metric Robustness**: Aggregated metrics like `Max_global` provide a more conservative and reliable indicator of model convergence compared to simple MSE or single-field errors.

## Related
- **Systems**: [[Viscoelastic_Fluids]]
- **Methods**: [[ViscoelasticNet]], [[Viscoelastic_Metrics]], [[Loss_History_Tracking]]
- **Topics**: [[Fluid_Dynamics]], [[Viscoelasticity]]
