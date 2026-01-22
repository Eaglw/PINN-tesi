# Track: Standardize Heat2D Point Distribution

## 1. Overview
This track aims to standardize the distribution and quantity of training points across all Heat2D experiments to ensure fair benchmarking. The goal is to define specific point sets for Internal Grid, Random Internal, and Boundaries, and apply them consistently across Neural Network (NN) and Physics-Informed Neural Network (PINN) cases as per the user's requirements.

## 2. Functional Requirements

### 2.1 Point Set Definitions
The system must generate three distinct sets of points initially:
1.  **Grid Points (Internal):** 1600 points evenly distributed on a grid strictly inside the domain.
    *   **Method:** `linspace(0, 1, 42)[1:-1]` for both x and y dimensions (resulting in 40x40 = 1600 points).
2.  **Random Points (Internal):** 1600 points randomly distributed strictly inside the domain.
    *   **Method:** Uniform sampling in (0, 1) x (0, 1).
3.  **Boundary Points:** 400 points uniformly distributed on the domain boundaries.
    *   **Method:** 100 points per side (Top, Bottom, Left, Right).

### 2.2 Experiment Case Configurations
The system must configure the four experiment cases to use the point sets as follows:

1.  **NN Random:**
    *   **Internal:** Uses the full set of **1600 Random Points**.
    *   **Boundary:** Uses the full set of **400 Boundary Points**.
    *   **Usage:** Standard Supervised Learning (MSE against Analytic Solution).

2.  **NN Grid:**
    *   **Internal:** Uses the full set of **1600 Grid Points**.
    *   **Boundary:** Uses the full set of **400 Boundary Points**.
    *   **Usage:** Standard Supervised Learning (MSE against Analytic Solution).

3.  **PINN Data+Physics:**
    *   **Physics Points (PDE Residual):** Uses the **1600 Grid Points**.
    *   **Boundary Conditions (BC Loss):** Uses the **400 Boundary Points**.
    *   **Data Points (Supervised Loss):** Uses a **subset of 1000 points** from the **1600 Random Points** set.
        *   *Note:* These 1000 points are used *only* for data loss, not physics.

4.  **PINN Pure Physics:**
    *   **Physics Points (PDE Residual):** Uses the **1600 Grid Points**.
    *   **Boundary Conditions (BC Loss):** Uses the **400 Boundary Points**.
    *   **Data Points:** None.

## 3. Implementation Details
*   **Consistency:** The random seed must be fixed (e.g., `torch.manual_seed(123)`) before generating the "Master Sets" of points to ensure the 1000 random points in Case 3 are indeed a subset of the 1600 in Case 1.
*   **Verification:** The script should print the shapes of the tensors for each case at the start of execution to verify the configuration.

## 4. Acceptance Criteria
*   **Case 1 (NN Random):** Training data size is exactly 2000 (1600 random + 400 boundary).
*   **Case 2 (NN Grid):** Training data size is exactly 2000 (1600 grid + 400 boundary).
*   **Case 3 (PINN Data+Phys):**
    *   Collocation points (Physics) = 1600 (Grid).
    *   BC points = 400.
    *   Supervised Data points = 1000 (Random subset).
*   **Case 4 (PINN Pure Phys):**
    *   Collocation points (Physics) = 1600 (Grid).
    *   BC points = 400.
*   **Code Quality:** No hardcoded magic numbers inside the training loops; configuration should be centralized.

## 5. Out of Scope
*   Changing the network architecture (layers, neurons).
*   Changing the optimization algorithms (Adam/L-BFGS settings).
*   Modifying the analytic solution logic.
