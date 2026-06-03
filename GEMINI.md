# PINN Tesi Project - Gemini Instructions

This project focuses on the research and application of Physics-Informed Neural Networks (PINNs) to solve viscoelastic fluid flows. It supports both direct problems (solving velocity, pressure, and stress fields) and inverse problems (identifying physical parameters like viscosities, relaxation time, and model parameters).

## Project Structure

- **`Viscoelastic/`**: Core module containing the viscoelastic fluid PINN implementation.
    - `Viscoelastic_main.py`: Entry point for running experiments (grid searches, forward, semi-inverse, or inverse solvers).
    - `results.csv`: Log file tracking performance metrics and physical parameters across runs.
    - `experiments_weighted/`: Output directory for generated logs, model checkpoints, and plots.
    - `src/`: Main source files:
        - `models.py`: Defines the Neural Network architectures (FCN and `ViscoelasticCombinedModel` which coordinates separate networks for $\psi$, $p$, and $\tau$).
        - `config.py`: Training parameters (`TrainingConfig`), learning rate schedulers, and network/parameter freezing helpers.
        - `Viscoelastic_physics.py`: Adimensional PDE physics constraints (momentum equations, constitutive equations for Oldroyd-B, PTT, Giesekus), and boundary condition rules.
        - `load_comsol.py`: Data loader and processing for COMSOL datasets.
        - `trainer.py`: Implementation of the staged optimization loop (Adam phase followed by L-BFGS refinement).
- **`COMSOL/`**: Storage folder for COMSOL reference datasets (e.g., `Oldroyd.csv`).
- **`func/`**: Shared utility functions.
    - `graphic_func.py`: Matplotlib plotting scripts for 2D visualizations, error maps, and comparisons.
    - `history_tracker.py`: Tracks and plots loss terms and parameter trajectories.
- **`models/`**: Directory for saving trained model states.
- **`plots/`**: Location for quick or general plots.

## Setup & Installation

### Virtual Environment
Use the standard virtual environment on Windows:
```powershell
python -m venv venv
.\venv\Scripts\activate
```

### Dependencies
Install dependencies from the workspace root:
```powershell
.\venv\Scripts\pip install -r requirements.txt
```
Key libraries: `torch`, `numpy`, `matplotlib`, `tqdm`, `pandas`.

## Running the Experiments

### Viscoelastic PINN Solver
Run the main grid search or configuration training:
```powershell
.\venv\Scripts\python Viscoelastic/Viscoelastic_main.py
```

## Development Conventions

### 1. Precision & Numerical Stability
- **Staged Precision Strategy**: 
    - Fase 1: Fast exploration with **Adam @ FP32** (leverages TF32 on Ampere GPUs).
    - Fase 2: Physical refinement with **L-BFGS @ FP64** for scientific-grade precision.
- **Default Type**: `torch.set_default_dtype(torch.float64)` is used during L-BFGS and final inference.

### 2. Architecture & Physics Standards
- **Stream-Function Formulation**: The network predicts the stream function $\psi$ instead of velocity components $u$ and $v$ directly, automatically satisfying the incompressibility constraint:
  $$u = \frac{\partial \psi}{\partial y}, \quad v = -\frac{\partial \psi}{\partial x}$$
- **Separate Network Heads**:
  - `model_psi` predicts stream function $\psi$ (dimension: 1 output).
  - `model_p` predicts pressure $p$ (dimension: 1 output).
  - `model_tau` predicts extra-stress tensor components $\tau = (\tau_{xx}, \tau_{xy}, \tau_{yy})$ (dimension: 3 outputs).
- **Constitutive Models**: Supports Oldroyd-B, PTT (Phan-Thien-Tanner), and Giesekus formulations through Weissenberg ($Wi$), Reynolds ($Re$), viscosity ratio ($\beta$), and model parameters ($\epsilon, \alpha$).

### 3. Staged Training Strategy (ViscoelasticNet Framework)
To ensure optimization stability, training is split into distinct stages:
1. **Phase 1 (Adam)**: Train only $\psi$ (velocity fields) and stress tensor $\tau$, while pressure $p$ is frozen.
2. **Phase 2 (Adam)**: Train $\psi$ and pressure $p$, keeping stress parameters adjusted or frozen depending on settings.
3. **Phase 3 (L-BFGS)**: Fine-tune the entire combined model jointly with FP64 precision.

## Note aggiunte
Prima di implementare o modificare effettivamente qualsiasi codice (escluse le letture, analisi del repo o prove innocue), spiegami sempre cosa stai cercando di fare. 
Se sei su windows non usare && per dare più comandi in uno, usa il modo corretto o runna singolarmente i comandi.
Su Windows, esegui SEMPRE i comandi python e pip facendo riferimento all'interprete del virtual environment (es. `.\venv\Scripts\python` o `.\venv\Scripts\pip`), senza dare per scontato che l'eseguibile globale sia presente nel PATH.
Per quanto riguarda l'analisi della repo, non modificare la struttura di staged training presente (con prima fase Adam solo psi e tau, poi psi e p, e fine L-BFGS con tutto acceso). Non proporre modifiche a questa struttura perché si desidera aderire rigorosamente al framework di viscoelasticnet. Inoltre, non passare mai i dati di stress provenienti da COMSOL alla PINN, poiché non avrebbe senso usare una PINN avendo già tutti i dati a disposizione.