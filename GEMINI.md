# PINN Tesi Project

This project focuses on the research and application of Physics-Informed Neural Networks (PINNs) to solve differential equations modeling physical systems. It supports both direct problems (finding the solution) and inverse problems (parameter identification).

## Project Structure

- **`PINNs_maurizio.py`**: Main script for the Damped Harmonic Oscillator problem.
- **`Harmonic oscillator PINN.ipynb`**: Jupyter notebook for the Harmonic Oscillator.
- **`IrreversibleCSTR/`**: Module for the Irreversible Continuous Stirred-Tank Reactor problem.
    - `IrreversibleCSTR_main.py`: Entry point for CSTR experiments.
    - `IrreversibleCSTR_inverse.py`: Focused on the inverse problem (parameter estimation).
- **`Heat2D/`**: Module for 2D Heat Transfer (Laplace equation).
    - `Heat2D_main.py`: Main script for 2D Heat experiments.
    - `Heat2D_prova.py`: Experimental/testing script for 2D Heat.
- **`Newtonian/`**: Module for Navier-Stokes equations and Newtonian fluids.
- **`notes/`**: Consolidated research documentation and technical journals.
- **`func/`**: Shared utility functions.
    - `graphic_func.py`: Plotting and GIF generation.
    - `history_tracker.py`: Loss history tracking and visualization.
- **`models/`**: Directory for saving trained models.
- **`plots/`** & **`Results/`**: Directories where generated plots and training artifacts are saved.

## Setup & Installation

### Virtual Environment
It is highly recommended to use a virtual environment.
- **MacOS/Linux**: Use the environment name `tesi` or `.venv`.
    ```bash
    python3 -m venv tesi
    source tesi/bin/activate
    ```
- **Windows**: Use `venv`.
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```

### Dependencies
Install the required Python packages:
```bash
pip install -r requirements.txt
```
Key dependencies: `torch`, `numpy`, `matplotlib`, `tqdm`, `Pillow`.

## Running the Experiments

### 1. Damped Harmonic Oscillator
Run the main script to train the PINN for the harmonic oscillator:
```bash
python PINNs_maurizio.py
```
Or explore the notebook:
```bash
jupyter notebook "Harmonic oscillator PINN.ipynb"
```

### 2. Irreversible CSTR
Run the CSTR simulation/inversion:
```bash
python IrreversibleCSTR/IrreversibleCSTR_main.py
```

### 3. 2D Heat Transfer
Run the 2D Heat transfer simulation:
```bash
python Heat2D/Heat2D_main.py
```

## Development Conventions

### 1. Precision & Numerical Stability
- **Staged Precision Strategy**: 
    - Fase 1: Esplorazione veloce con **Adam @ FP32** (sfrutta TF32 su GPU Ampere).
    - Fase 2: Raffinamento fisico con **L-BFGS @ FP64** per precisione "scientific grade".
- **Default Type**: `torch.set_default_dtype(torch.float64)` è lo standard per la fase di raffinamento e inferenza finale.

### 2. Architecture Standards
- **Tapered Layers**: Utilizzo di architetture a imbuto (es. `[120, 100, 80, 60, 40, 20]`) per condensare le feature.
- **Activations**: 
    - **SiLU (Swish)**: Preferita per PDE di secondo ordine grazie alla regolarità della derivata seconda.
    - **LAA (Learnable Adaptive Activations)**: Implementazione di parametri scalabili per catturare gradienti ripidi.

### 3. Sampling & Training
- **Sobol sequences**: Utilizzo di campionamento quasi-Monte Carlo per una copertura uniforme del dominio.
- **SAR (Spatially Adaptive Refinement)**: Aggiunta dinamica di punti nelle zone con alto residuo PDE.

### 4. Visualization & Output
- **Visualizzazione**: Funzioni centralizzate in `func/graphic_func.py`.
- **Output**: Controllo e creazione automatica di `Results/` e `plots/`.
- **Device**: Utilizzo di `cuda` con fallback su `cpu`.

## Note aggiunte
Prima di scrivere qualsiasi tipo di codice spiegami cosa stai cercando di fare. 
Se sei su windows non usare && per dare più comandi in uno, usa il modo corretto o runna singolarmente i comandi