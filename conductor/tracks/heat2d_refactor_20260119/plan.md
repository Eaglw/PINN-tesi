# Implementation Plan - Refactor Heat2D

## Phase 1: Analysis and Scaffolding [checkpoint: 9a4e774]
- [x] Task: Analyze `Heat2D` module structure and dependencies
    - [x] Read `Heat2D/Heat2D_PINN.py` and `Heat2D/Heat2D_main.py`
    - [x] Identify hardcoded PDE residuals and boundary conditions
- [x] Task: Set up testing infrastructure
    - [x] Create `tests/` directory
    - [x] Configure `pytest` and `coverage`
    - [x] Create a dummy test to verify configuration
- [x] Task: Define `PhysicsProblem` Interface
    - [x] Create `Heat2D/physics.py`
    - [x] Define abstract base class `PhysicsProblem` with methods like `residual(x, u)` and `boundary_condition(x, u)`
- [x] Task: Conductor - User Manual Verification 'Analysis and Scaffolding' (Protocol in workflow.md)

## Phase 2: Modular Physics Implementation [checkpoint: 3835446]
- [x] Task: Implement `HeatEquation2D` Class
    - [x] Create `tests/test_physics_heat2d.py`
    - [x] Write tests for the `residual` calculation (using known analytical solutions or simple checks)
    - [x] Implement `HeatEquation2D` inheriting from `PhysicsProblem`
    - [x] Port logic from `Heat2D_PINN.py` to `HeatEquation2D`
- [x] Task: Refactor `PINN` Class
    - [x] Create `tests/test_pinn_solver.py`
    - [x] Write tests for PINN initialization and forward pass
    - [x] Modify `PINN` class in `Heat2D/Heat2D_PINN.py` to accept a `PhysicsProblem` instance
    - [x] Replace hardcoded PDE logic with calls to `physics_problem.residual`
- [x] Task: Conductor - User Manual Verification 'Modular Physics Implementation' (Protocol in workflow.md)

## Phase 3: Integration and Verification
- [x] Task: Update Main Script
    - [x] Modify `Heat2D/Heat2D_main.py` to instantiate `HeatEquation2D` and pass it to the solver
    - [x] Ensure all hyperparameters and configurations are preserved
- [x] Task: Verify Reproducibility
    - [x] Run the refactored `Heat2D_main.py`
    - [x] Compare results (loss curves, final plots) with baseline results (from `Results/` or previous runs)
- [x] Task: Expand Test Coverage
    - [x] Run coverage report
    - [x] specific tests for any uncovered lines (e.g., edge cases in boundary conditions)
    - [x] Ensure >80% coverage
- [~] Task: Conductor - User Manual Verification 'Integration and Verification' (Protocol in workflow.md)
