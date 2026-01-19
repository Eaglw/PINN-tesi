# Track Specification: Refactor Heat2D for Modularity and Testability

## Goal
The primary goal of this track is to refactor the existing `Heat2D` module to decouple the physics definition (the PDE) from the solver logic (the PINN training loop). This separation will facilitate the introduction of new physical models (e.g., Non-Newtonian fluids) in future tracks. Additionally, the track aims to establish a robust unit testing foundation, targeting >80% code coverage, to ensure reliability and prevent regressions.

## Scope
-   **Analyze Existing Code:** Review `Heat2D/Heat2D_PINN.py`, `Heat2D/Heat2D_NN.py`, and `Heat2D/Heat2D_main.py` to identify coupling points.
-   **Define Interfaces:** Create abstract base classes or protocols for `PhysicsProblem` and `PINNSolver`.
-   **Implement Modular Physics:** Extract the 2D Heat Equation logic into a dedicated class implementing `PhysicsProblem`.
-   **Refactor Solver:** Update the PINN solver to accept a `PhysicsProblem` instance rather than having hardcoded equations.
-   **Add Tests:** Implement unit tests for the new physics class, the refactored solver, and utility functions using `pytest`.
-   **Verify:** Ensure that the refactored code reproduces the results of the original implementation.

## Out of Scope
-   Implementing new physics (e.g., Navier-Stokes) in this track.
-   Refactoring the `IrreversibleCSTR` module (this will be a separate track).
-   Major changes to the visualization or results logging infrastructure (unless strictly necessary for the refactor).

## Success Criteria
-   **Modularity:** The PDE definition is separated from the training loop.
-   **Test Coverage:** Code coverage for the refactored `Heat2D` components exceeds 80%.
-   **Reproducibility:** The refactored code produces results consistent with the original baseline (within numerical tolerance).
-   **Clean Code:** The code adheres to the project's Python style guide.
