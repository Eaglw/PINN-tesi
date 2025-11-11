# Project Overview

This project is a collection of Python scripts and Jupyter notebooks for exploring and applying Physics-Informed Neural Networks (PINNs). The primary focus is on solving differential equations that model physical systems. The project demonstrates the use of PINNs for both direct and inverse problems.

The main technologies used are:
- Python
- PyTorch
- NumPy
- Matplotlib

The project is structured around two main examples:
1.  **Damped Harmonic Oscillator:** Explored in `PINNs_maurizio.py` and `Harmonic oscillator PINN.ipynb`. These files cover various aspects of PINNs, including:
    - Training a standard neural network.
    - Solving the direct problem (finding the solution given the parameters).
    - Solving the inverse problem (finding the physical parameters of the system).
    - Experimenting with different optimizers (Adam, LBFGS) and activation functions (Tanh, GELU).
2.  **Irreversible Continuous Stirred-Tank Reactor (CSTR):** Implemented in the `IrreversibleCSTR` module. This example demonstrates how to apply PINNs to a chemical engineering problem, both with and without training data.

The `func` directory contains helper functions, such as `graphic_func.py` for plotting results and generating GIFs of the training process.

# Building and Running

## Dependencies

The project's dependencies are listed in `requirements.txt`. It is recommended to use a virtual environment.

To install the dependencies, run:
```bash
pip install -r requirements.txt
```

## Running the Scripts

The Python scripts can be run directly from the command line. For example:

```bash
python PINNs_maurizio.py
```

or

```bash
python IrreversibleCSTR/IrreversibleCSTR_main.py
```

The scripts are configured to run specific experiments based on the `goal` variable inside the script. You may need to edit the script to choose which experiment to run.

## Running the Jupyter Notebook

The `Harmonic oscillator PINN.ipynb` notebook can be run using a Jupyter server:

```bash
jupyter notebook "Harmonic oscillator PINN.ipynb"
```

# Development Conventions

- The code is organized into a combination of standalone scripts and modules.
- Helper functions, especially for plotting, are kept in the `func` directory.
- The scripts and notebooks save plots and GIFs to the `plots` directory to visualize the results and the training progress.
- The code is well-commented, with explanations of the different PINN concepts and implementation details.
- The project uses a `.gitignore` file to exclude common Python artifacts and environment directories from version control.
