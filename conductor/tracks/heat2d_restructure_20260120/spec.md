# Specification: Heat2D Module Restructuring and Experiment Organization

## Overview
Reorganize the `Heat2D/` directory to improve readability and maintainability. The core logic (source code) will be separated from the experiment execution and artifacts. Results will be organized into goal-oriented folders within an `experiments/` directory, each containing the necessary code, documentation, and visual artifacts for reproducibility.

## Functional Requirements
- **Directory Structure Refactoring:**
    - Create `Heat2D/src/` to house core training logic and physics definitions (e.g., `physics.py`, `Heat2D_NN.py`, `Heat2D_PINN.py`).
    - Rename/Structure the `Results/` directory into `Heat2D/experiments/`.
- **Goal-Oriented Organization:**
    - Organize `Heat2D/experiments/` into subfolders categorized by their numeric goal (e.g., `0_NN_Classic/`, `1_PINN_DataPhys/`, `2_Pure_Physics/`).
- **Experiment Artifacts:**
    - Each experiment folder MUST contain:
        - A `plots/` subdirectory for all images and GIFs.
        - A **copy of the training script** used for that specific run.
        - A `README.md` or a header in the script providing metadata (run description, hyperparameters, artifact guide).
- **Script Documentation (Headers):**
    - Training scripts within the experiment folders must include a header comment block describing:
        - The specific goal of the run.
        - A guide to the produced files (Artifact Guide).
        - Contextual notes for subdirectories (e.g., explaining `high_res` vs `std_res`).
- **Main Entry Point:**
    - `Heat2D/Heat2D_main.py` remains in the `Heat2D/` root and will be updated to orchestrate runs using the new structure.

## Non-Functional Requirements
- **Reproducibility:** Ensuring that each experiment folder contains the exact code used to generate its results.
- **Readability:** Reducing clutter in the `Heat2D/` root directory.
- **Consistency:** Applying the numeric categorization consistently with the `GOAL` variable in the main script.

## Acceptance Criteria
1. The `Heat2D/` root contains only `Heat2D_main.py` and essential auxiliary scripts.
2. Core logic is moved to `Heat2D/src/`.
3. Running a specific goal in `Heat2D_main.py` correctly populates `Heat2D/experiments/<Goal_Name>/` with plots and the documented script copy.
4. Experiment scripts contain the requested explanatory headers.
5. Existing experiments (like `optim_collocation`) are migrated to the new structure with appropriate documentation for their sub-structures.

## Out of Scope
- Restructuring modules other than `Heat2D` (e.g., `IrreversibleCSTR`).
- Changing the underlying physics or model architectures (purely structural refactoring).
