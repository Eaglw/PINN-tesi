# Product Definition

## Initial Concept
Il caso del CSTR e dello scambio di calore sono due esempi che mi servono per settare la codebase del mio progetto di tesi. Una volta rifiniti ed esplorati questi due esempi, il repo si concentrerà su PINN applicate a problemi di CFD, in particolare qualche problema benchmark su fluidi non newtoniani. Il fine ultimo è quello di avere tutti gli strumenti necessari a portare avanti lo sviluppo di PINN applicate alla CFD.

## Target Audience
- **Researcher (Thesis Author):** The primary user is the thesis author, requiring a flexible and reliable experimentation platform.
- **Academic Advisors/Reviewers:** Secondary users who may review the code for reproducibility and method validation.
- **Future Researchers:** Those extending the work on Non-Newtonian CFD PINNs.

## Core Goals
- **Foundational Validation:** Use CSTR and 2D Heat Transfer problems to validate the PINN architecture (Pure Physics vs Data-Driven), training strategies (e.g., warm-up, hybrid optimization), and code structure.
- **CFD Readiness:** Establish a codebase capable of scaling to Computational Fluid Dynamics problems, specifically for Non-Newtonian fluids.
- **Modular Physics:** Decouple the solver logic from the physical equations to allow seamless transitions from ODEs (CSTR) / simple PDEs (Heat) to complex systems (Navier-Stokes + Constitutive equations).
- **Inverse Problem Capability:** Ensure robust functionality for parameter estimation (e.g., identifying viscosity coefficients or reaction rates) alongside forward simulations.

## Key Features
- **Modular Physics Engine:** A clear interface for defining differential equations, boundary conditions, and domain geometry, allowing new physics (like Non-Newtonian rheology) to be plugged in without rewriting the training loop.
- **Hybrid Optimization Pipeline:** Built-in support for switching between optimizers (Adam, L-BFGS) and strategies (warm-up phases) to handle stiff or complex problems.
- **Unified Experimentation:** Centralized logging of metadata and performance metrics (CSV format), history tracking, and visualization tools to enable systematic comparison across different architectures and strategies.
- **Visual Analytics:** Standardized visualization pipelines generating unified 2x2 comparison grids and high-precision error maps, enabling immediate visual assessment of model performance across different architectures and training strategies.
- **Reproducible Workflows:** Automatic snapshotting of training scripts and configuration for every experiment run, ensuring full reproducibility of results.
- **Inverse Solving:** dedicated workflows for identifying unknown physical parameters from sparse data.
- **Pure Physics Solving:** Dedicated mode for solving forward problems using only physics residuals and boundary conditions (no experimental data).
