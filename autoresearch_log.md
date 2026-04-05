# Autoresearch Log - Heat2D Mini L2 Optimization

**Goal**: Minimize `L2_Relative_Error` for the `heat2dmini` setup.
**Metric**: `L2_Relative_Error` (lower is better).
**Scope**: `heat2dmini/Heat2D_adaptive_mini.py`, `Heat2D/src/Heat2D_PINN.py`.
**Verification Command**: `.\venv\Scripts\python.exe heat2dmini/verify_metric_fast.py`.

## Theoretical Background
The project uses Physics-Informed Neural Networks (PINNs) to solve the 2D Heat equation (Laplace) on a square domain. The solution is compared against an analytical series expansion. Key components include:
- **Adaptive Activations**: Learnable slope parameters for each layer to improve gradient capture.
- **Dynamic Weighting**: Learning Rate Annealing to balance boundary and physics losses.
- **Tapering Architecture**: A strategy to optimize network capacity versus training stability.

## Research History Summary

### Phase 1: Foundation (Iter 0-10)
Discovery that SiLU and tapering architectures are significantly better than flat ones with Tanh or GELU.

### Phase 2: Capacity Expansion (Iter 11-20)
Introduction of extra wide layers and increased training epochs (2500 Adam). Reached L2 < 0.008.

### Phase 3: Precision Refinement (Iter 21-33)
Optimization of L-BFGS iterations (settled on 1500) and investigation of adaptive activation initializations. Reached the current best of **0.00680**.

### Phase 4: Consolidation (Current)
Consolidated results from various independent test series into a unified workflow to avoid redundant experiments.

## Current Best Configuration (Iter 24)
- **Architecture**: `ADAPTIVE_[120, 120, 100, 80, 60, 40, 20]`
- **Activation**: SiLU (Adaptive)
- **Optimization**: 2500 Adam + 1500 L-BFGS
- **BC Weight**: 25.0
- **Sampling**: Sobol (1600 points, 0.02 margin)

## Iteration 41: Periodic Resampling
- **Hypothesis**: Il ricampionamento periodico dei punti di collocazione durante la fase Adam migliora la generalizzazione della PDE.
- **Result**: **Discard** (L2: 0.00709). Il ricampionamento ogni 500 epoche non ha superato il record di 0.00680. È possibile che la variazione frequente dei punti impedisca ad Adam di stabilizzarsi prima della fase L-BFGS.

## Iteration 42: L-BFGS History Size Expansion
- **Hypothesis**: Aumentare l' `history_size` di L-BFGS da 300 a **500** permette un raffinamento più accurato.
- **Result**: **Discard** (L2: 0.007749). L'espansione della memoria non ha portato benefici, suggerendo che 300 sia già sufficiente o che la stabilità sia compromessa da una memoria troppo lunga.

## Iteration 43: Final Layer Capacity Expansion
- **Hypothesis**: Aumentare la larghezza dell'ultimo layer nascosto da 20 a **40** (`120, 120, 100, 80, 60, 40, 40`).
- **Result**: **Discard** (L2: 0.00816). L'aumento della capacità finale ha regredito la precisione, probabilmente introducendo complessità inutile o rendendo la fase finale meno stabile.

## Iteration 44: Accelerated Loss Weight Adaptation
- **Result**: **Discard** (L2: 0.007491). L'aggiornamento ogni 50 epoche non ha superato il record, suggerendo che la stabilità dei pesi ogni 100 epoche sia preferibile per la convergenza di Adam.

## Iteration 45: Boundary Constraint Strengthening
- **Result**: **Discard** (L2: 0.008152). L'aumento della risoluzione al contorno non ha migliorato il record, suggerendo che 400 punti totali siano già sufficienti.

## Iteration 46: Initial Layer Capacity Expansion
- **Hypothesis**: Aumentare il primo layer a 150 neuroni.
- **Result**: **Discard** (L2: 0.011778). Il peggioramento conferma l'efficacia del tapering a 120.

## Iteration 47: Tighten L-BFGS Tolerances
- **Result**: **Discard** (L2: 0.011703). Il peggioramento è stato causato da una baseline errata (architettura a 150 neuroni ereditata per errore). Dopo il reset del codice alla configurazione 120 (Iter 24), procediamo con nuovi test.

## Iteration 48: Extended Adam Pre-training
- **Hypothesis**: Aumentare le epoche Adam da 2500 a **2700**.
- **Motivation**: Un periodo più lungo di pre-ottimizzazione con Adam può aiutare la rete ad attraversare regioni piatte o rumorose della loss, raggiungendo un bacino di attrazione più profondo prima che L-BFGS prenda il controllo per il raffinamento finale.
- **Expected Result**: L2 Error < 0.00680.
