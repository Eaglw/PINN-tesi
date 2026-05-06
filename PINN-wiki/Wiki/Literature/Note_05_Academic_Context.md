# Note: Academic Context (Safa Jamali)

- **Type**: Research Summary
- **Subject**: ML in Chemical Engineering & Rheology
- **Reference**: `Reference/Note_mio_studio/05_Academic_Context_Safa_Jamali.md`

## Summary
Deep dive into the application of Machine Learning in Chemical Engineering, specifically focusing on complex fluids (RhINNs) and the research of Prof. Safa Jamali.

## Key Pillars
1. **Flow Modeling**: Using RhINNs to simulate non-Newtonian fluids with sparse data.
2. **Automated Experimentation**: High-throughput robotics integrated with Active Learning.
3. **Material Characterization**: Mapping microstructural states (granular packing) to macro properties (viscosity).
4. **Model Discovery**: Using SINDy-like techniques to discover governing equations.

## Multi-Fidelity Networks (MFNN)
A two-stage strategy to optimize simulation costs:
- **Low-Fidelity (LF)**: Trained on large, approximate datasets to learn the general trend.
- **High-Fidelity (HF)**: Trained on sparse, expensive experimental data to learn the correction to the LF model.

## Future Directions
- **Neural Operators**: DeepONet/FNO for geometry-agnostic modeling.
- **Memory Effects**: Capturing viscoelasticity via LSTM or fractional derivatives.

## Related
- **Topics**: [[PINN_Fundamentals]], [[Fluid_Dynamics]]
- **Systems**: [[Fluid_Dynamics]]
- **Methods**: [[Tapered_Architectures]]
