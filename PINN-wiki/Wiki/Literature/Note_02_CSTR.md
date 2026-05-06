# Note: CSTR Research Journal

- **Type**: Technical Journal
- **Subject**: Irreversible CSTR Modeling & Optimization
- **Reference**: `Reference/Note_mio_studio/02_CSTR_Research_Journal.md`

## Summary
This journal documents the systematic experimentation framework for the Irreversible Continuous Stirred-Tank Reactor (CSTR) problem. It tracks the evolution from simple FCN baselines to advanced coupled multi-network architectures.

## Key Findings
- **Hybrid Optimization**: The most effective strategy involves an initial phase with **Adam** for broad convergence, followed by **L-BFGS** for high-precision refinement.
- **Activation Functions**: **GELU** combined with hybrid optimization shows superior performance compared to the Tanh baseline.
- **Coupled PINN Architecture**: For non-isothermal reactors, using two specialized networks (**ConcentrationNet** and **TemperatureNet**) coupled via physical balance residuals is necessary.
- **Warm-up Strategy**: To prevent numerical explosion in the Arrhenius term, a 1000-step warm-up phase (where physical loss is zero) is critical to stabilize the temperature range.

## Related
- **Systems**: [[CSTR_Modeling]]
- **Methods**: [[Staged_Precision_Strategy]], [[Tapered_Architectures]]
- **Topics**: [[PINN_Fundamentals]], [[Activation_Functions]]
