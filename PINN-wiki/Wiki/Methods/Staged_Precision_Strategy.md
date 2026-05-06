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
Switching requires converting both the model and the tensors to double precision:
```python
torch.set_default_dtype(torch.float64)
model.double()
x_train = x_train.double()
```
Additionally, `torch.backends.cuda.matmul.allow_tf32` should be set to `False` during the FP64 phase to ensure full precision.

## References
- Core strategy defined in [[Note_01_Framework]].
