# Specification: Heat2D Precision Sensitivity Benchmarking

## Overview
Develop a rigorous benchmarking framework for the `heat2d-fp64-32` codebase to mathematically evaluate the impact of `float32` vs `float64` across all components of the PINN training loop. The goal is to perform an exhaustive combinatorial analysis to identify which parts strictly require high precision and which can be optimized without exceeding a 1-2% error threshold.

## Functional Requirements
1. **Granular Component Toggling**:
   - Refactor the code to allow independent precision control (FP32 vs FP64) for the following "Toggleable Parts":
     - **Part A: Neural Network** (Weights, Biases, and Activations).
     - **Part B: Data Loss** (MSE calculation against experimental/ground truth data).
     - **Part C: Physics Residuals** (Automatic Differentiation and Laplacian/PDE terms).
     - **Part D: Boundary/Initial Condition Loss**.
     - **Part E: Optimizer State & Accumulation** (Gradients and update steps, especially in L-BFGS).
2. **Combinatorial Benchmark Runner**:
   - Create a script (`exhaustive_precision_benchmark.py`) that executes training for all $2^N$ combinations of the toggleable parts.
   - Each run must use the same random seeds and initialization to isolate the precision impact.
3. **Rigorous Evaluation Metrics**:
   - For each combination, capture:
     - **Physical Error**: MAE and Max Absolute Error relative to the "Gold Standard" (Full FP64 run).
     - **Numerical Stability**: Condition numbers or gradient norm variance.
     - **Performance**: Execution time and peak GPU/CPU memory usage.
4. **Sensitivity Report & Decision Matrix**:
   - Automatically generate a report (CSV/Plot) that ranks each component by its "Precision Sensitivity".
   - Identify the "Pareto Optimal" configuration: Highest performance with <2% physical error.

## Non-Functional Requirements
- **Directory Isolation**: All changes must be strictly contained within the `heat2d-fp64-32` directory.
- **Reproducibility**: Strict seeding of all random components (NumPy, Torch, CUDA).

## Acceptance Criteria
- [ ] Framework supports independent precision toggling for all 5 identified parts.
- [ ] Benchmarking script executes all combinations automatically.
- [ ] Sensitivity report identifies exactly where FP64 is mathematically necessary to maintain accuracy.
- [ ] Final results confirm which mixed-precision configuration is the most efficient.

## Out of Scope
- Half-precision (FP16/BF16) or TensorFloat-32 (TF32) hardware-specific features.
- Modifying the core physics or architecture, only the precision of operations.
