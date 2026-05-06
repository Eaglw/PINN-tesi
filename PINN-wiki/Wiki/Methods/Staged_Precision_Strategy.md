# Method: Staged Precision Strategy

A two-phase training approach to balance speed and scientific accuracy.

## Phase 1: Fast Exploration
- **Optimizer**: Adam
- **Precision**: FP32 (utilizing TF32 on Ampere GPUs)
- **Goal**: Rapid hyperparameter search and initial convergence.
- **Speedup**: Measured at 10x-12x vs FP64.

## Phase 2: Physical Refinement
- **Optimizer**: L-BFGS
- **Precision**: FP64 (`float64`)
- **Goal**: "Scientific grade" accuracy; eliminating high-frequency residuals.

## Implementation Details
Switching requires converting the model and data: `model.to(torch.float64)`.

## References
- Core strategy defined in [[Note_01_Framework]].
