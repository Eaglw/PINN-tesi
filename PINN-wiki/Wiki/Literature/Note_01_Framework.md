# Literature: Note 01 Framework

- **Source**: `Reference/Note_mio_studio/01_PINN_Implementation_Framework.md`
- **Type**: Internal Research Note
- **Key Themes**: [[Tapered_Architectures]], [[Activation_Functions]], [[Sampling_Strategies]], [[Staged_Precision_Strategy]].

## Summary
Central technical reference for PINN implementation. Covers architectural choices like funnel structures, adaptive activations (LAA), and comparative analysis of Tanh vs GELU vs SiLU. It introduces the "Staged Precision Strategy" for hybrid training (FP32/FP64) and sampling techniques like Sobol and SAR.

## Key Takeaways
- **Funnel Structure**: `[120, 100, 80, 60, 40, 20]` reduces overfitting.
- **LAA**: $f(x) = \sigma(a \cdot x)$ for local gradient scaling.
- **Hybrid Precision**: Adam @ FP32 (Exploration) -> L-BFGS @ FP64 (Refinement).
- **Sobol**: Essential for uniform domain coverage.
