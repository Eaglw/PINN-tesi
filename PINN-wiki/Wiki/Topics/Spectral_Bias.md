---
title: "Spectral Bias / Frequency Principle"
---

## Overview
**Spectral Bias** (or **Frequency Principle**) is a fundamental phenomenon in Deep Learning where neural networks exhibit a preference for learning low-frequency components of a target function before capturing high-frequency details. This has significant implications for Physics-Informed Neural Networks (PINNs), as many physical systems involve multiscale phenomena or high-frequency gradients that can be slow to converge.

## Theoretical Basis
The regularity of the activation function directly impacts the decay rate of the loss function in the Fourier domain. High-frequency components are "penalized" by the smooth nature of standard activations (like Tanh or Sigmoid), leading to slower learning of fine-grained residuals.

## Technical Implementation in PINNs
To address spectral bias in this project, several strategies can be employed:
- **Adaptive Activation Functions**: Using trainable scaling parameters to adjust the slope of activations, effectively allowing the network to "tune" into higher frequencies.
- **Multi-scale Architectures**: Utilizing different subnetworks or input scaling (e.g., Fourier Features) to explicitly represent different frequency scales.
- **Staged Training**: Fitting coarse features first and then refining with higher precision or localized sampling.

## References
- [[Frequency_Spectral_Bias]]
- [[Sharma_et_al_Hyperparameter_Selection]]
