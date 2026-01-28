# Heat 2D Comprehensive Audit Report

## 2. Metric Dictionary & Code Consistency Audit

This section rigorously defines the metrics used to evaluate the PINN and standard NN models, providing mathematical formulas and verifying their implementation in the codebase.

### Evaluation Metrics

| Metric | Formula | Code Reference | Description |
| :--- | :--- | :--- | :--- |
| **L2 Relative Error** | $\frac{\|T_{pred} - T_{true}\|_2}{\|T_{true}\|_2}$ | `func/logging_utils.py` -> `compute_metrics` | Global norm-based error measuring the overall reconstruction quality. |
| **Max Relative Error Peak** | $\max \left( \frac{\|T_{pred} - T_{true}\|}{\|T_{true}\|} \right) \times 100$ | `func/logging_utils.py` -> `compute_metrics` | Pointwise maximum percentage error, masked for $|T_{true}| > 0.01$ to avoid singularity at boundaries. |
| **PDE Residual Loss** | $\frac{1}{N_{phys}} \sum \|T_{xx} + T_{yy}\|^2$ | `Heat2D/src/Heat2D_PINN.py` -> `heat2d_physics_loss` | Measures how well the neural network satisfies the Laplace equation. |
| **Boundary Loss (BC)** | $\frac{1}{N_{bc}} \sum \|T_{pred} - T_{bc}\|^2$ | `func/history_tracker.py` -> `compute_pinn_loss` | Measures adherence to Dirichlet boundary conditions. |

### Consistency Audit Results

| Component | Status | Finding |
| :--- | :--- | :--- |
| **Relative Error Calculation** | **PASS** | `graphic_func.py` and `logging_utils.py` were updated (Commit `d4de304`) to use local relative error with a $0.01$ threshold mask, replacing the previous global-max normalization. |
| **Logging Schema** | **PASS** | `results.csv` correctly tracks `Loss_Weight`, `n_points`, and `Run_Type`, allowing for fair comparison across grid-searches. |
| **Data Integrity** | **PASS** | `history_tracker.py` handles `None` values during warmup phases correctly, preventing data misalignment in loss plots. |
| **Training Consistency** | **PASS** | `Heat2D_main.py` uses "Master Sets" (1600 points) and fixed seeds (`123`) to ensure reproducibility across all 4 benchmark goals. |
