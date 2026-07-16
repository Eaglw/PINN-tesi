# PINN Tesi Project - Gemini Instructions

This project focuses on the research and application of Physics-Informed Neural Networks (PINNs) to solve viscoelastic fluid flows. It supports both direct problems (solving velocity, pressure, and stress fields) and inverse problems (identifying physical parameters like viscosities, relaxation time, and model parameters).

## Project Structure

- **`final_roll/`**: Active working folder containing the current viscoelastic fluid PINN implementation for the four-roll mill.
    - `train_4roll_main.py`: Main entry point for training and running current experiments.
    - `src/`: Core source code module:
        - `train.py`: Implementation of the model architecture (`CombinedModel`) and the training process.
        - `physics.py`: Viscoelastic physics definition, including losses, adimensional parameters, and boundary conditions.
        - `utils.py`: Utility functions for loading COMSOL data and plotting fields.
        - `debug.py`: Helper tools for debugging magnitudes and random point evaluations.
    - `output_4rollmill/`: Directory where results, plots, logs, and models are saved.
- **`Prep-tests/`**: Archival directory containing old tests, code drafts, and backups (e.g., Newtonian flows, 2D Heat equations, and previous Viscoelastic PINN experiments).
- **`scratch/`**: Workspace folder dedicated to quick tests, exploratory scripts, and temporary experiments. Definitive and production code should not reside here.
- **`COMSOL/`**: Storage folder for COMSOL reference datasets (e.g., `4roll/4_roll_mill.csv`).

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

### Viscoelastic PINN Solver (4-Roll Mill)
Run the main script from the `final_roll` folder or set the `PYTHONPATH` environment variable:
```powershell
# Option 1: Navigate to final_roll and run the script
cd final_roll
..\venv\Scripts\python train_4roll_main.py
cd ..

# Option 2: Set PYTHONPATH from the root directory
$env:PYTHONPATH="final_roll;."
.\venv\Scripts\python final_roll/train_4roll_main.py
```

## Development Conventions

### 1. File Placement & Workflow
- **`final_roll/`**: Only clean, production-ready, or verified modifications to the core PINN script and components must be checked into this directory.
- **`scratch/`**: All random tests, trial-and-error scripts, temporary plots, and draft implementations must be kept in this directory. 
- **`Prep-tests/`**: Contains historical, deprecated, or archived tests. Do not work directly in this folder.

### 2. Precision & Numerical Stability
- **Staged Precision Strategy**: 
    - Fase 1: Fast exploration with **Adam @ FP32** (leverages TF32 on Ampere GPUs).
    - Fase 2: Physical refinement with **L-BFGS @ FP64** for scientific-grade precision.
- **Default Type**: `torch.set_default_dtype(torch.float64)` is used during L-BFGS and final inference.

### 3. Architecture & Physics Standards
- **Stream-Function Formulation**: The network predicts the stream function $\psi$ instead of velocity components $u$ and $v$ directly, automatically satisfying the incompressibility constraint:
  $$u = \frac{\partial \psi}{\partial y}, \quad v = -\frac{\partial \psi}{\partial x}$$
- **Separate Network Heads**:
  - `model_psi` predicts stream function $\psi$ (dimension: 1 output).
  - `model_p` predicts pressure $p$ (dimension: 1 output).
  - `model_tau` predicts extra-stress tensor components $\tau = (\tau_{xx}, \tau_{xy}, \tau_{yy})$ (dimension: 3 outputs).
- **Constitutive Models**: Supports Oldroyd-B, PTT (Phan-Thien-Tanner), and Giesekus formulations through Weissenberg ($Wi$), Reynolds ($Re$), viscosity ratio ($\beta$), and model parameters ($\epsilon, \alpha$).

### 4. Staged Training Strategy (ViscoelasticNet Framework)
To ensure optimization stability, training is split into distinct stages:
1. **Phase 1 (Adam)**: Train only $\psi$ (velocity fields) and stress tensor $\tau$, while pressure $p$ is frozen.
2. **Phase 2 (Adam)**: Train $\psi$ and pressure $p$, keeping stress parameters adjusted or frozen depending on settings.
3. **Phase 3 (L-BFGS)**: Fine-tune the entire combined model jointly with FP64 precision.

## Note aggiunte
Prima di implementare o modificare effettivamente qualsiasi codice (escluse le letture, analisi del repo o prove innocue), spiegami sempre cosa stai cercando di fare. 
Se sei su windows non usare && per dare più comandi in uno, usa il modo corretto o runna singolarmente i comandi.
Su Windows, esegui SEMPRE i comandi python e pip facendo riferimento all'interprete del virtual environment (es. `.\venv\Scripts\python` o `.\venv\Scripts\pip`), senza dare per scontato che l'eseguibile globale sia presente nel PATH.
Inoltre, non passare mai i dati di stress provenienti da COMSOL alla PINN, poiché non avrebbe senso usare una PINN avendo già tutti i dati a disposizione.