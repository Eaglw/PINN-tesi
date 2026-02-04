# Track: Heat2D Point Overlap Prevention

### Overview
This track aims to ensure numerical stability and training consistency in the `Heat2D` module by preventing any spatial overlap between different sets of training points: collocation points (physics), internal data points, and boundary data points. Overlapping points can lead to redundant gradients or conflicting loss contributions, especially in high-precision scientific computing.

### Functional Requirements
- **Boundary Safety Margin:** Implement a fixed epsilon margin ($\epsilon = 10^{-5}$) for all internal point generation (grid and random) to ensure no internal point ever lies exactly on a boundary.
- **Set Disjointness:** Implement a filtering mechanism using a Euclidean distance threshold ($d_{min} = 10^{-4}$) to ensure that no two points from different functional sets (e.g., collocation vs. data) are spatially too close.
- **Target Count Integrity:** If filtering reduces the number of points below the requested target, the system must iteratively regenerate and filter additional points until the exact target count is achieved.
- **Grid & Random Support:** The exclusion logic must be robust for both grid-based and random-based sampling strategies in the domain.
- **Boundary Conditions (BC) Integrity:** Ensure boundary points are strictly on the edges and do not overlap with internal collocation or data points (guaranteed by the safety margin).

### Technical Constraints
- Use `torch` operations for distance calculations and filtering to maintain compatibility with GPU acceleration.
- The default precision must remain `torch.float64`.

### Acceptance Criteria
- [ ] Internal collocation points and internal data points have a minimum distance of $10^{-4}$.
- [ ] No internal point (collocation or data) is within $10^{-5}$ of the domain boundaries.
- [ ] The final number of points in each set matches the user-defined hyperparameters (e.g., 1600 collocation points, 1000 data points).
- [ ] Visual verification: Plots of point distributions show no overlapping markers.

### Out of Scope
- Optimization of the filtering algorithm for extremely large datasets (millions of points).
- Changes to the physical equations or model architecture.
