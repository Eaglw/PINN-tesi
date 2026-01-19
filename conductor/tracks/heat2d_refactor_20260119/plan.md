# Implementation Plan - Refactor Heat2D

## Phase 1: Analysis and Scaffolding
- [ ] Task: Analyze `Heat2D` module structure and dependencies
    - [ ] Read `Heat2D/Heat2D_PINN.py` and `Heat2D/Heat2D_main.py`
    - [ ] Identify hardcoded PDE residuals and boundary conditions
- [ ] Task: Set up testing infrastructure
    - [ ] Create `tests/` directory
    - [ ] Configure `pytest` and `coverage`
    - [ ] Create a dummy test to verify configuration
- [ ] Task: Define `PhysicsProblem` Interface
    - [ ] Create `Heat2D/physics.py`
    - [ ] Define abstract base class `PhysicsProblem` with methods like `residual(x, u)` and `boundary_condition(x, u)`
- [ ] Task: Conductor - User Manual Verification 'Analysis and Scaffolding' (Protocol in workflow.md)

## Phase 2: Modular Physics Implementation
- [ ] Task: Implement `HeatEquation2D` Class
    - [ ] Create `tests/test_physics_heat2d.py`
    - [ ] Write tests for the `residual` calculation (using known analytical solutions or simple checks)
    - [ ] Implement `HeatEquation2D` inheriting from `PhysicsProblem`
    - [ ] Port logic from `Heat2D_PINN.py` to `HeatEquation2D`
- [ ] Task: Refactor `PINN` Class
    - [ ] Create `tests/test_pinn_solver.py`
    - [ ] Write tests for PINN initialization and forward pass
    - [ ] Modify `PINN` class in `Heat2D/Heat2D_PINN.py` to accept a `PhysicsProblem` instance
    - [ ] Replace hardcoded PDE logic with calls to `physics_problem.residual`
- [ ] Task: Conductor - User Manual Verification 'Modular Physics Implementation' (Protocol in workflow.md)

## Phase 3: Integration and Verification
- [ ] Task: Update Main Script
    - [ ] Modify `Heat2D/Heat2D_main.py` to instantiate `HeatEquation2D` and pass it to the solver
    - [ ] Ensure all hyperparameters and configurations are preserved
- [ ] Task: Verify Reproducibility
    - [ ] Run the refactored `Heat2D_main.py`
    - [ ] Compare results (loss curves, final plots) with baseline results (from `Results/` or previous runs)
- [ ] Task: Expand Test Coverage
    - [ ] Run coverage report
    - [ ] specific tests for any uncovered lines (e.g., edge cases in boundary conditions)
    - [ ] Ensure >80% coverage
- [ ] Task: Conductor - User Manual Verification 'Integration and Verification' (Protocol in workflow.md)
