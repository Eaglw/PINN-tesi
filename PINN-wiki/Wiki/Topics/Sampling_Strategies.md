# Topic: Sampling Strategies

Sampling is the process of selecting points in the domain where the PDE residual (Physical Loss) is evaluated.

## Quasi-random Sampling (Sobol/Halton)
Unlike uniform random sampling, low-discrepancy sequences like **Sobol** and **Halton sequences** are designed to cover the domain more uniformly.
- **Benefit**: Faster convergence of the training loss and better accuracy with fewer points (as discussed in [[Sharma_et_al_Hyperparameter_Selection]]).
- **Halton Sequence**: Specifically effective for boundary and domain sampling in heat transfer problems.

## Spatially Adaptive Refinement (SAR)
A dynamic sampling method that increases point density in regions with high PDE residuals.
- **Goal**: Focused learning on high-error areas.

## Management of Overlaps and Boundaries
- **Duplicate Prevention**: Use `torch.unique` on boundary points.
- **Distance Check**: Verify minimum distance (e.g., \(10^{-4}\)) using `torch.cdist`.
- **Safety Margin**: Synchronize internal point generation with boundary margins.

## References
- See [[Note_01_Framework]] for implementation details.
