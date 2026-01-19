# Product Guidelines

## Documentation Style
- **Physics Rationale:** Every new physical module (e.g., Navier-Stokes, Non-Newtonian constitutive models) must be accompanied by an external Markdown file in `notes/` detailing the equations, assumptions, and validation metrics.
- **Code Clarity:** Use clear variable naming for physical parameters (e.g., `viscosity`, `reynolds_number`). Complex tensor operations, particularly automatic differentiation calls (`torch.autograd.grad`), must have inline comments explaining which derivative is being computed and why.
- **Reference Management:** Maintain links between code implementations and specific literature or PDF references (e.g., those in `Reference/`).

## Organization & Output
- **Experiment Hierarchy:** Organise results hierarchically: `Results/<Problem_Type>/<Specific_Experiment>/`.
- **Run Identification:** Use timestamped or uniquely ID'd subfolders for each training session to prevent overwriting results and ensure reproducibility.
- **Performance Logging:** Every significant experiment should update a centralized log or summary table (e.g., within `notes/`) to track progress across different architectures (activation functions, optimizer strategies).

## Development Workflow
- **Precision:** Default to `torch.float64` for all scientific calculations to avoid accumulation of numerical errors in stiff physical systems.
- **Modularization:** Physics definitions should be separated from the neural network architecture. The goal is to keep `PINN` classes as generic as possible.
- **Visualization:** Use the centralized `func/graphic_func.py` for all plotting to maintain a consistent visual style suitable for inclusion in the final thesis.
