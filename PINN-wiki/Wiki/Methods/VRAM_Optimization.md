# VRAM Optimization

## Overview
Training Physics-Informed Neural Networks (PINNs) on complex systems such as 2D Viscoelastic fluid flows (combining Navier-Stokes and Oldroyd-B constitutive equations) requires the evaluation of high-order spatial derivatives via automatic differentiation (`torch.autograd.grad` with `create_graph=True`). This generates massive computational graphs that accumulate significant memory pressure, frequently leading to `CUDA out of memory` (OOM) errors on consumer-grade GPUs with limited VRAM (e.g., 4GB NVIDIA GTX 1050 Ti).

VRAM Optimization encompasses a suite of software engineering and algorithmic techniques designed to minimize peak memory allocation, prevent VRAM fragmentation, and enable full-dataset, high-precision (FP64) training without hardware upgrades.

## Technical Implementation

In the Viscoelastic PINN codebase (`Viscoelastic/`), VRAM optimization is achieved through four specialized pillars:

### 1. Dynamic Weighting Optimization (Eliminating Redundant Graphs)
In the original implementation of Wang et al.'s Learning Rate Annealing algorithm, the dynamic loss weights ($\lambda$) were updated every 100 epochs by executing secondary forward passes and autograd graph constructions on the entire dataset (5000 internal points and 800 boundary points). 
- **Optimization**: The redundant full-batch evaluations were removed entirely. The dynamic weighting logic now extracts the unweighted loss components directly from the cached `loss_dict` generated during the primary mini-batch forward pass. This maintains identical mathematical stability while eliminating periodic memory spikes.

### 2. PyTorch CUDA Allocator Configuration
Long-running grid search experiments often suffer from VRAM fragmentation, where sufficient total memory is available but split into non-contiguous blocks.
- **Optimization**: The environment variable `os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"` is injected at the initialization of `Viscoelastic_main.py`. This instructs the PyTorch CUDA allocator to manage memory via expandable virtual memory segments, significantly reducing fragmentation overhead.

### 3. L-BFGS Optimizer Tuning (`history_size`)
The L-BFGS optimizer is critical for scientific-grade precision refinement in FP64. However, it maintains an internal history buffer of past parameter steps ($s_k$) and gradient differences ($y_k$).
- **Optimization**: For a combined multi-network architecture ($\psi, p, \tau$) with ~350,000 parameters in FP64, the default `history_size=300` allocates **~1.68 GB of VRAM** purely for the optimizer state. A dynamic hardware detection mechanism inspects `torch.cuda.get_device_properties(0).total_memory`. If a GPU with $\le 4.5$ GB VRAM is detected, `history_size` is automatically scaled down to `50` (reducing state memory to ~280 MB), instantly freeing **1.4 GB of VRAM**.

### 4. Gradient Accumulation (Chunking) in L-BFGS Closure
To achieve true full-batch L-BFGS convergence in FP64 without exceeding memory limits, the 5000 collocation points cannot be evaluated simultaneously.
- **Optimization**: Gradient accumulation (Chunking) is implemented inside the L-BFGS `closure()`. When the total number of points exceeds `chunk_size` (dynamically set to `500` for 4GB GPUs), the dataset is split into chunks of 500 points. For each chunk, the loss is computed, scaled by `1/num_chunks`, and backpropagated (`loss_chunk.backward()`). PyTorch accumulates the exact parameter gradients and instantly frees the autograd graph before processing the next chunk. This achieves 100% mathematically identical full-batch FP64 gradients while capping peak graph VRAM at the footprint of only 500 points.

### 5. Final Loss Check Optimization
Following L-BFGS completion, logging the final loss value previously required a full-batch evaluation outside the closure, triggering an immediate OOM.
- **Optimization**: `last_loss_val` and `last_loss_dict` containers track the exact loss scalar and dictionary during the final L-BFGS closure evaluation. The post-training check directly retrieves these cached values, bypassing graph construction entirely.

## References
- [[Dynamic_Weighting]]: Learning Rate Annealing methodology.
- [[Staged_Precision_Strategy]]: Transitioning from FP32 Adam to FP64 L-BFGS.
- [[GPU_Optimization]]: Complementary performance and synchronization tuning.
- [[Viscoelastic_Training]]: Main experiment guide for viscoelastic fluid flows.
