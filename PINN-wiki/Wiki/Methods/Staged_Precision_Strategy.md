# Method: Staged Precision Strategy

A two-phase numerical strategy to balance exploration speed and scientific accuracy.

## Phase 1: Fast Exploration
- **Optimizer**: Adam
- **Precision**: Strict FP32 (with **TF32 explicitly disabled**)
- **Goal**: Rapid hyperparameter optimization, basin discovery, and kinematic convergence.
- **Speedup**: Measured at 3x-5x vs FP64 on standard CUDA GPUs.

> [!WARNING]
> **Global TF32 Deactivation**:
> On NVIDIA Ampere/Ada GPUs, TensorFloat-32 (TF32) is enabled by default in PyTorch for matrix multiplications. While fast, TF32 truncates the mantissa from 23 bits to just **10 bits** (equivalent to FP16 precision with FP32 exponent range).
> In PINNs, computing high-order spatial derivatives via Autograd (especially $\nabla^2 \mathbf{u}$ and constitutive gradients) compounds this truncation error, introducing an artificial numerical noise floor of $\sim 10^{-3}$ that stalls convergence.
> Therefore, **TF32 must be globally deactivated from the very start of training**:
> ```python
> torch.backends.cuda.matmul.allow_tf32 = False
> torch.backends.cudnn.allow_tf32 = False
> ```

## Phase 2: Physical Refinement
- **Optimizer**: L-BFGS
- **Precision**: Scientific-grade FP64 (`float64`)
- **Goal**: Elimination of high-frequency residuals, exact satisfaction of constitutive equations, and precise parameter identification.

## Implementation Details
Switching from Adam FP32 to L-BFGS FP64 requires converting the model, physics parameters, data tensors, and setting the default PyTorch dtype:
```python
torch.set_default_dtype(torch.float64)
model.to(torch.float64)
physics.to(torch.float64)
# Recursive verification of data dictionary via assert_fp64_integrity
```

## References
- [[Numerical_Hygiene_and_Phase2_Reforms]] (Comprehensive analysis of TF32 noise)
- [[Viscoelastic_Training]]
- [[Note_01_Framework]]
