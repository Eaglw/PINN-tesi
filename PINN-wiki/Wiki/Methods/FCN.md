# Method: Fully Connected Network (FCN)

## Overview
A Fully Connected Network (FCN), also commonly referred to as a Multi-Layer Perceptron (MLP), is the foundational neural network architecture used in Physics-Informed Neural Networks (PINNs). In an FCN, neurons in each layer are connected to all neurons in the subsequent layer through learnable weight matrices and bias vectors. 

For 2D physical problems, an FCN typically maps spatial coordinates $(x, y)$ (and optionally time $t$) to continuous physical fields (e.g., stream function $\psi$, pressure $p$, stress tensor $\boldsymbol{\tau}$). It serves as a universal function approximator whose spatial and temporal derivatives can be computed exactly and analytically via automatic differentiation (`autograd`), allowing the network to be directly constrained by the governing partial differential equations (PDEs) [[Note_01_Framework]].

## Technical Implementation
Within the PINN-tesi project, FCNs are deployed across multiple physical systems with tailored architectural configurations:

- **Viscoelastic Flow ([[Viscoelastic_Training]])**: The multi-network architecture (`ViscoelasticCombinedModel`) decouples the physical fields into distinct FCN sub-networks: `model_psi` (scalar stream function $\psi$), `model_p` (scalar pressure $p$), and `model_tau` (stress tensor components $\tau_{xx}, \tau_{xy}, \tau_{yy}$). These FCNs leverage [[Tapered_Architectures]] (funnel-style configurations such as `[2, 120, 100, 80, 60, 40, 20, 1]`) and `nn.SiLU` ([[Activation_Functions]]) to guarantee smooth, continuous second-order derivatives required for the Navier-Stokes momentum equations [[Thakur_et_al_ViscoelasticNet]].
- **Harmonic Oscillator ([[Harmonic_Oscillator]])**: An FCN maps time $t$ to displacement $u(t)$ using `nn.GELU` activations and L-BFGS optimization to solve both direct and inverse vibration problems [[Maurizio_Harmonic_Oscillator]].
- **2D Heat Transfer ([[Heat2D_Analysis]])**: An FCN maps $(x,y)$ to temperature $T(x,y)$ to solve the steady-state Laplace equation across the spatial domain [[Note_03_Heat2D]].

## References
- [[Note_01_Framework]]
- [[Thakur_et_al_ViscoelasticNet]]
- [[Maurizio_Harmonic_Oscillator]]
- [[Note_03_Heat2D]]
