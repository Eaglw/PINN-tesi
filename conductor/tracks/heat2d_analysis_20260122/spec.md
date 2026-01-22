# Specification: Heat2D Results Analysis Script

## Overview
This track involves creating a standalone Python script, `Heat2D/analyze_results.py`, designed to process `Heat2D/results.csv` and generate comprehensive visualizations for training performance analysis. The tool will enable systematic comparison of different PINN and NN architectures, hyperparameters, and training strategies used in the 2D Heat Transfer problem.

## Functional Requirements
- **Data Loading:** Read `Heat2D/results.csv` and handle potential missing values or data type conversions.
- **Key Metric Analysis:** Prioritize `Max_Relative_Error_Peak` as the primary performance indicator.
- **Secondary Metric Analysis:** Include `L2_Relative_Error`, `Loss_Total`, `Loss_Physics`, and `Loss_Data` in the analysis.
- **Comparison Dimensions:**
    - **Methodology:** Compare `Run_Type` (e.g., `NN_Random`, `NN_Grid`, `PINN_PurePhys`, `PINN_HardBC`).
    - **Architecture & Activation:** Compare different hidden layer configurations and activation functions (`Tanh`, `GELU`).
    - **Training Depth:** Analyze the effect of `Epochs` on accuracy.
- **Visualizations:**
    - **Bar Charts:** Comparative performance of different methods for key metrics.
    - **Scatter Plots:** Correlation analysis (e.g., `Loss_Total` vs `Max_Relative_Error_Peak`).
    - **Box Plots:** Distribution of errors across multiple seeds to assess stability.
    - **Heatmaps:** Performance matrix across hyperparameters (e.g., Architecture vs Activation).
- **Output:** Save generated plots as high-resolution images in a dedicated directory (e.g., `Heat2D/analysis_plots/`).

## Non-Functional Requirements
- **Standalone Execution:** The script should be runnable manually via `python Heat2D/analyze_results.py`.
- **Readability:** Use `pandas` for data manipulation and `matplotlib`/`seaborn` for professional-grade plotting.
- **Extensibility:** Structure the code to allow easy addition of new metrics or plot types.

## Acceptance Criteria
- [ ] Script successfully parses `results.csv`.
- [ ] Generates at least one of each plot type (Bar, Scatter, Box, Heatmap) covering the requested dimensions.
- [ ] Plots clearly label axes, legends, and titles for interpretability.
- [ ] Summary statistics or conclusions are printed to the console or saved to a log.

## Out of Scope
- Real-time training monitoring (handled by existing `history_tracker.py`).
- Automated model re-training or hyperparameter optimization.
