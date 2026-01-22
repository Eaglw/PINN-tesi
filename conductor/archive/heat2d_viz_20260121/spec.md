# Specification - Heat2D Visualization Improvements

## Overview
This track aims to refine and standardize the visualization output for Heat2D experiments. The focus is on making the final result plots more concise and improving the comparison across different model types (Neural Networks vs. Physics-Informed Neural Networks) by centralizing error maps and including hyperparameter metadata.

## Functional Requirements

### 1. Refined Final Result Plot (`final_result.png`)
- **Layout:** Change from the current 3-column layout (Solution, Absolute Error, Relative Error) to a 2-column layout.
- **Content:**
    - **Left Plot:** Solution $u(x, y)$. This plot must now overlay the training points.
        - **Physics Collocation Points:** Plotted with a distinct color.
        - **Data Points:** Plotted with a different distinct color.
        - **Legend:** A clear legend must be included to differentiate the two sets of points.
    - **Right Plot:** Relative Error Map.
- **Exclusion:** The Absolute Error map will be removed from this specific summary plot.

### 2. Unified Comparison Plot
- **Layout:** Create a 2x2 grid of error maps to compare different training strategies.
- **Subplots:**
    - (0,0): NN with Random sampling.
    - (0,1): NN with Grid sampling.
    - (1,0): PINN with Data + Physics.
    - (1,1): PINN with Pure Physics.
- **Colorbars:** Each subplot will have its own individual colorbar to highlight the error distribution within that specific model.
- **Metadata Title:** The main title of the comparison figure must include the experiment's hyperparameters:
    - Network Architecture (e.g., layers/neurons).
    - Number of Epochs.
    - Activation Function.

## Non-Functional Requirements
- **Consistency:** Ensure consistent styling (font sizes, colormaps) across both the final result and comparison plots.
- **Modular Integration:** The changes should be implemented within the `Heat2D` module, ideally leveraging or extending existing functions in `func/graphic_func.py` if applicable, without breaking other modules.

## Acceptance Criteria
- [ ] `Heat2D` training scripts generate a 2-plot `final_result.png` (Solution with points + Relative Error).
- [ ] A new comparison script or function generates a 2x2 grid of error maps for the four specified models.
- [ ] Comparison plots include hyperparameters in the main title.
- [ ] Training points in the solution plot are visually distinguishable and labeled in a legend.

## Out of Scope
- Modifications to the CSTR or Damped Harmonic Oscillator visualization logic.
- Changes to the underlying training logic or loss functions.
