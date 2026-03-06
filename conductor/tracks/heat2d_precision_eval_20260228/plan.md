# Implementation Plan: Heat2D Precision Sensitivity Benchmarking

## Phase 1: Component Mapping & Architecture Refactor
- [x] Task: Audit `heat2d-fp64-32/src/` to isolate the 5 identified toggleable parts (NN, Data Loss, Physics, BC, Optimizer).
- [x] Task: Refactor the model and training loop to accept a "Precision Configuration" object (e.g., a bitmask or dict).
- [x] Task: Implement a "Precision-Aware Cast" wrapper that handles `to(torch.float32)` and `to(torch.float64)` based on the config.
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Component Mapping & Architecture Refactor' (Protocol in workflow.md)

## Phase 2: Combinatorial Runner Development
- [x] Task: Implement `exhaustive_precision_benchmark.py` to iterate through all precision combinations.
- [x] Task: Integrate standardized logging for error metrics (MAE vs Full FP64) and performance (Time/Memory).
- [x] Task: Ensure the "Gold Standard" (Full FP64) is executed first and its results are cached for comparison.
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Combinatorial Runner Development' (Protocol in workflow.md)

## Phase 3: Sensitivity Analysis & Execution
- [~] Task: Execute the exhaustive benchmark suite (32 combinations for 5 parts).
- [x] Task: Implement a visualization script to generate heatmaps or bar charts showing the "Error vs Speedup" trade-off.
- [ ] Task: Analyze results to identify the components that cause the largest error spikes when switched to FP32.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Sensitivity Analysis & Execution' (Protocol in workflow.md)

## Phase 4: Final Validation & Decision Documentation
- [ ] Task: Verify directory isolation (no changes outside `heat2d-fp64-32`).
- [ ] Task: Create a comprehensive report summarizing which parts *must* stay in FP64 and which can be FP32.
- [ ] Task: Clean up the refactored code to ensure it remains readable while supporting the optimal mixed-precision config.
- [ ] Task: Conductor - User Manual Verification 'Phase 4: Final Validation & Decision Documentation' (Protocol in workflow.md)
