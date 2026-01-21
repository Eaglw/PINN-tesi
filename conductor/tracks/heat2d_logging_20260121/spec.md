# Specification: Heat2D Experiment Logging System

## Overview
This track implements a centralized logging system for the Heat2D forward simulation experiments. It aims to capture comprehensive training metadata and performance metrics in a structured CSV format, enabling easy comparison between different neural network architectures and PINN strategies directly within VS Code.

## Functional Requirements
1. **Global Results Storage:**
   - Create and maintain a `Heat2D/results.csv` file.
   - If the file doesn't exist, it must be initialized with a header row.
   - New results must be appended as rows.

2. **Logging Columns (Ordered):**
   - `Timestamp`: Date and time of the experiment completion.
   - `Architecture`: Description of the layers and neurons (e.g., "[2, 50, 50, 50, 50, 1]").
   - `Activation_Func`: The activation function used (e.g., "Tanh", "GELU").
   - `Epochs`: Total training iterations.
   - `Run_Type`: The specific experiment case (e.g., "NN_Random", "NN_Grid", "PINN_DataPhys", "PINN_PurePhys").
   - `Optimizer`: Name of the optimizer (Adam, L-BFGS, or Hybrid).
   - `Learning_Rate`: The initial or main learning rate used.
   - `Loss_Total`: Final total loss value.
   - `Loss_Physics`: Final physics residual loss.
   - `Loss_Boundary`: Final boundary condition loss.
   - `Loss_Data`: Final data-driven loss (where applicable).
   - `L2_Relative_Error`: Global L2 relative error norm.
   - `Max_Relative_Error_Peak`: The maximum pointwise relative error value across the domain.
   - `Seed`: The random seed used for the run.

3. **Implementation Details:**
   - Define reusable functions `compute_metrics` and `update_results_csv` within a new file `func/logging_utils.py`.
   - `compute_metrics` will calculate L2 Relative Error and Pointwise Max Relative Error using the model, grid, and analytical solution.
   - `update_results_csv` will handle file creation/appending with proper CSV formatting.
   - Call `update_results_csv` at the conclusion of each training loop in `Heat2D/Heat2D_main.py`.

## Non-Functional Requirements
- **Readability:** CSV format chosen for compatibility with VS Code "Edit CSV" or "Excel Viewer" extensions.
- **Precision:** Log numerical values with scientific notation or high precision (float64) to ensure accuracy.
- **Robustness:** Ensure logging doesn't crash the main training loop if file access fails (e.g., file open by another process).

## Acceptance Criteria
- [ ] `Heat2D/results.csv` is created or updated after running `Heat2D_main.py`.
- [ ] All requested metadata and metrics are correctly recorded in the specified order.
- [ ] Performance metrics (L2 Error, Max Peak, Partial Losses) match the values computed during the run.
- [ ] The file remains readable and correctly formatted after multiple consecutive runs.

## Out of Scope
- Logging for inverse problems (`Heat2D_main_inverse.py`).
- Automated plotting of the CSV data.
- Cloud or database integration.
