# Sharma et al. - Hyperparameter selection for physics-informed neural networks (PINNs) – Application to discontinuous heat conduction problems

- **Type**: Journal Article (Numerical Heat Transfer, Part B: Fundamentals)
- **Date**: October 9, 2023 (Online) / 2024 (Volume 85)
- **Authors**: Prakhar Sharma, Llion Evans, Michelle Tindall, Perumal Nithiarasu
- **Reference**: [[Selezione Iperparametri PINNs.pdf]]

## Summary
This paper provides a systematic evaluation of PINN hyperparameters focusing on steady-state heat conduction problems with **discontinuities** in boundary conditions or coefficients. The study identifies optimal ranges for depth, width, learning rate, and activation functions through extensive manual tuning and comparison.

## Key Hyperparameter Findings
- **Activation Functions**:
  - **Best Performers**: **SiLU** and **GELU** showed the most consistent accuracy and stability.
  - **Avoid**: **ReLU** (vanishing gradients) and **SELU** (exploding gradients in deep PINNs).
  - **Tanh/Sin**: Effective but can be sensitive to initialization and high learning rates in stiff problems.
- **Network Depth**: Accuracy generally increases with depth. **20 hidden layers** were found optimal for capturing complex gradients, provided sufficient training iterations are used.
- **Learning Rate**: Avoid large learning rates (e.g., \(1 \times 10^{-2}\)) in stiff or discontinuous problems, as they lead to convergence anomalies. Optimal range: \(5 \times 10^{-4}\) to \(7.5 \times 10^{-3}\).
- **Architecture**: Standard Feedforward (FCNN) outperformed Modified Fourier Neural Networks (MFNN) for these specific discontinuous problems due to MFNN's susceptibility to spectral bias in some contexts.

## Advanced Techniques
- **Signed Distance Function (SDF)**: Crucial for problems with discontinuous BCs. SDFs weight points near discontinuities less, preventing the optimizer from getting stuck on sharp transitions.
- **Integral Loss Formulation**: Scaling loss terms by the domain/boundary volume (length/area) ensures that term contributions are balanced, preventing interior points from overwhelming the loss.
- **Quasi-random Sampling**: Using **Halton sequences** instead of uniform random sampling leads to faster convergence and better domain coverage with fewer points.
- **Adaptive Activation**: Employing trainable coefficients at the neuron level accelerates convergence.

## Analyzed Systems
- 2D and 3D steady-state heat conduction with discontinuous BCs.
- Parametric conductivity and parametric geometry.
- Discontinuous conductivity (piecewise functions).

## Related
- **Topics**: [[Activation_Functions]], [[Sampling_Strategies]], [[PINN_Fundamentals]]
- **Methods**: [[SDF_for_Discontinuities]], [[Integral_Loss_Scaling]]
- **Systems**: [[Heat2D_Analysis]]
