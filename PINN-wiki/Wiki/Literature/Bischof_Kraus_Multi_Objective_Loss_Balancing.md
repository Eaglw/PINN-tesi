# Multi-Objective Loss Balancing for Physics-Informed Deep Learning (Bischof & Kraus)

## Summary
- **Authors**: Rafael Bischof, Michael A. Kraus (Swiss Data Science Center / ETH Zürich, 2021/2022).
- **Core Focus**: Investigates multi-objective loss balancing methods for Physics-Informed Neural Networks to prevent gradient pathologies and training failures caused by conflicting loss magnitudes across PDEs, boundary conditions, and data measurements.
- **Proposed Methodology**: Proposes **ReLoBRaLo** (Relative Loss Balancing with Random Lookback), a self-adaptive, gradient-free loss weighting algorithm, and benchmarks it against Learning Rate Annealing, GradNorm, and SoftAdapt across forward and inverse PDE problems (Burgers, Kirchhoff plate bending, and Helmholtz).

---

## Key Methodology

### 1. Multi-Objective Formulation of PINNs
- Formulates PINN training as a linear scalarized multi-objective optimization problem:
  $$ \mathcal{L}(\boldsymbol{\theta}, \boldsymbol{\mu}) = \sum_{i=1}^m \lambda_i(t) \mathcal{L}_i(\boldsymbol{\theta}, \boldsymbol{\mu}) $$
- Distinguishes between gradient-based adaptive schemes (GradNorm, Learning Rate Annealing) requiring multiple backward passes and loss-history-based adaptive schemes (SoftAdapt, ReLoBRaLo).

### 2. ReLoBRaLo Algorithm Mechanics
- **Relative Progress Balancing**: Computes scaling factors based on the relative improvement of each loss term between time steps $t'$ and $t$ using a softmax with temperature $\mathcal{T}$:
  $$ \lambda_i^{\text{bal}}(t, t') = m \cdot \frac{\exp\left( \frac{\mathcal{L}_i(t)}{\mathcal{T} \mathcal{L}_i(t')} \right)}{\sum_{j=1}^m \exp\left( \frac{\mathcal{L}_j(t)}{\mathcal{T} \mathcal{L}_j(t')} \right)} $$
- **Saudade (Random Lookback)**: Introduces a Bernoulli random variable $\rho$ ($\mathbb{E}[\rho] \approx 0.999 - 1.0$) to occasionally evaluate progress against the initial loss $\mathcal{L}_i(0)$, helping the network escape local minima:
  $$ \lambda_i^{\text{hist}}(t) = \rho \lambda_i(t-1) + (1 - \rho) \lambda_i^{\text{bal}}(t, 0) $$
- **Exponential Moving Average (EMA)**:
  $$ \lambda_i(t) = \alpha \lambda_i^{\text{hist}}(t) + (1 - \alpha) \lambda_i^{\text{bal}}(t, t-1) $$
  with smoothing factor $\alpha \in [0.9, 0.999]$.

### 3. Comparison with Other Adaptive Balancing Schemes
- **Learning Rate Annealing (Wang et al.)**: Uses gradient norm ratios; highly effective for inverse parameter identification but incurs $\mathcal{O}(m)$ autograd overhead.
- **GradNorm (Chen et al.)**: Explicitly balances gradient norms and relative training rates w.r.t. a shared layer; high computational overhead ($+70\%$ time).
- **SoftAdapt (Heydari et al.)**: Normalizes loss slopes without long-term history or lookback.
- **ReLoBRaLo**: Zero extra backward passes ($\sim 40\%$ faster than LR Annealing, $\sim 70\%$ faster than GradNorm), matching or exceeding baseline accuracy.

---

## Key Findings & Project Relevance

- **Computational Efficiency**: Demonstrates that loss-based relative balancing eliminates multiple backward passes, saving significant training time and VRAM.
- **Inverse Problem Insight**: Confirms that while ReLoBRaLo excels at forward multiscale problems with low overhead, Learning Rate Annealing converges faster for low-dimensional inverse parameter estimation ($\boldsymbol{\mu}$).
- **Validation of Adaptive Loss Weighting**: Highlights why static loss weighting fails on coupled PDE systems and provides alternative strategies for multi-field balancing.

---

## Related Concepts
- **Topics**: [[Loss_Functions]], [[PINN_Fundamentals]], [[Inverse_Problems]], [[EMA_Smoothing]]
- **Methods**: [[Dynamic_Weighting]], [[GPU_Optimization]], [[VRAM_Optimization]], [[Viscoelastic_Residual_Scaling]]
- **Systems**: [[Viscoelastic_Fluids]], [[Viscoelastic_Training]], [[Harmonic_Oscillator]], [[CSTR_Modeling]]
