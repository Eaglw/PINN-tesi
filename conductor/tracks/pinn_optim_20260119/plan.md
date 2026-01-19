# Implementation Plan - PINN Performance Analysis and Optimization

## Phase 1: Comparative Analysis
- [ ] Task: Baseline Comparison Run
    - [ ] Run both NN (Grid) and PINN with 10,000 epochs and identical architectures.
    - [ ] Compare spatial relative error maps side-by-side using `plot_error_map_comparison`.
- [ ] Task: Gradient Analysis
    - [ ] Instrument the PINN training loop to log the norms of gradients for `bc_loss`, `data_loss`, and `pde_loss`.
    - [ ] Generate a plot showing the magnitude of these gradients over the first 5,000 epochs.
- [ ] Task: Document Findings
    - [ ] Summarize whether physics gradients are conflicting with data gradients or if the pde_loss is just stagnating.

## Phase 2: Targeted Optimizations
- [ ] Task: Implement Loss Balancing Improvements
    - [ ] Update `train_modelPINN` to support configurable weights for each loss term (increasing $\lambda_{bc}$ and $\lambda_{data}$ if they are being dominated).
    - [ ] Test a run with higher weight on boundaries.
- [ ] Task: Enhance Collocation Strategy
    - [ ] Increase the number of collocation points in `train_modelPINN` (e.g., from 50x50 to 100x100).
    - [ ] Verify if higher resolution physics reduces the error map peaks.
- [ ] Task: Conductor - User Manual Verification 'Targeted Optimizations' (Protocol in workflow.md)

## Phase 3: Final Verification
- [ ] Task: Unified Performance Test
    - [ ] Run the optimized PINN for the full 30,000 epochs.
    - [ ] Compare the final relative error map with the original baseline.
- [ ] Task: Cleanup and Documentation
    - [ ] Remove any temporary logging or debug plots.
    - [ ] Update project notes with the optimization results.
- [ ] Task: Conductor - User Manual Verification 'Final Verification' (Protocol in workflow.md)
