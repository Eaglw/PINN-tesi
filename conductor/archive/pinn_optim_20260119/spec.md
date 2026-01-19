# Track Specification: PINN Performance Analysis and Optimization

## Overview
The user is unsatisfied with the final relative error of the PINN model in the `Heat2D` problem, which remains high even after extensive training. This track focuses on analyzing the root causes of this underperformance, particularly in comparison to the standard Neural Network (NN) without physics, and implementing targeted improvements.

## Functional Requirements
- **Analysis Phase:**
    - Conduct a comparative analysis between the PINN and the standard NN (Grid-based) using identical architectures.
    - Analyze the spatial distribution of the error (Error Maps) to identify if the issue is localized (e.g., boundaries vs. center).
    - Investigate Gradient Magnitudes: Log and compare the gradients of the Loss components (Data vs. Physics) to check for "Gradient Pathologies" (e.g., physics gradients dominating or vanishing).
- **Optimization Phase (Hypothesis-Driven):**
    - **Experiment A (Loss Balancing):** Implement dynamic or tuned loss weighting (e.g., increasing the weight of the Data/BC term relative to the Physics term).
    - **Experiment B (Collocation Points):** Increase the density of collocation points or change their distribution to see if the physics resolution is the bottleneck.
- **Reporting:**
    - Generate a summary report (markdown or plots) linking the analysis findings to the proposed fixes.

## Non-Functional Requirements
- **Modularity:** Implement any new loss weighting or gradient logging logic as reusable components in `func/`.
- **Reproducibility:** Maintain fixed seeds for all experiments.

## Acceptance Criteria
- Analysis confirms whether the issue is optimization-based (gradients) or capacity-based.
- At least one optimization strategy (Loss Balancing or Collocation) is implemented and tested.
- A final comparison plot demonstrates whether the relative error has decreased compared to the baseline PINN.

## Out of Scope
- Changing the fundamental PDE (Heat Equation).
- switching to entirely different network architectures (e.g., CNNs, Transformers).
