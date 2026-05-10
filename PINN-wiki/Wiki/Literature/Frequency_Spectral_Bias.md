---
title: "Frequency principle/spectral bias"
source: "[Frequency principlespectral bias.md](file:///c:/Users/eaglw/Documents/PINN%20tesi/PINN-wiki/Reference/Frequency%20principlespectral%20bias.md)"
author: "Wikipedia"
type: "clipping"
---

## Summary
The **frequency principle** (or **spectral bias**) describes the tendency of deep neural networks (DNNs) to fit target functions from low to high frequencies during training. This means that low-frequency components (coarse features) are learned faster than high-frequency components (fine details/noise).

## Key Methodology
- **Observation**: Robustly observed in DNNs regardless of overparametrization.
- **Mechanism**: The regularity of the activation function translates into the decay rate of the loss function in the frequency domain.
- **Analysis Tools**: Discrete Fourier Transform (DFT), projection methods, Gaussian filters.

## Key Findings
- DNNs are good at learning low-frequency functions but struggle with high-frequency ones.
- **Algorithms to overcome spectral bias**:
    - **PhaseDNN**: Converts high frequencies downward for learning.
    - **Adaptive Activation Functions**: Modifies scaling factors to accelerate convergence (e.g., Jagtap et al.).
    - **Multi-scale DNN (MscaleDNN)**: Uses scaling coefficients to handle multiple frequency ranges.
    - **Fourier Feature Networks**: Maps inputs to high-frequency features (sines/cosines).
    - **Multi-stage Neural Networks (MSNN)**: Superposition of DNNs fitting residuals.
- **Early-stopping**: Can be used to avoid learning high-frequency noise.

## Related
- [[Activation_Functions]]
- [[PINN_Fundamentals]]
- [[Tapered_Architectures]]
