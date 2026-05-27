# Method: GPU Bottleneck Optimization

Techniques and analysis to maximize GPU utilization, specifically for small Physics-Informed Neural Networks (PINNs) trained on powerful hardware (e.g., RTX 3080).

## Overview
When training small networks on a very fast GPU, the computational time for the forward and backward passes can be extremely low (microseconds). In these scenarios, the overall training speed is often bounded by the **CPU-GPU overhead** and **implicit synchronizations**, rather than the raw compute power (FLOPS) of the GPU.

## Key Bottlenecks Identified

### 1. Implicit Synchronization via `.item()`
Calling `.item()` on a PyTorch tensor forces the CPU to wait for the GPU to finish all pending operations, then copies a single scalar value across the PCIe bus from VRAM to RAM.
- **Problem**: If `total_loss.item()`, `pde_loss.item()`, etc., are called *every epoch* (e.g., inside a `history_tracker.py`), it completely destroys asynchronous CUDA execution. The GPU sits idle waiting for the CPU to issue the next command.
- **Solution**: Accumulate tensors in a list and move them to the CPU in blocks (e.g., every 100 epochs), or use `.detach()` to store tensors without gradients and only call `.item()` when printing or plotting.

### 2. Control Flow Synchronization (`torch.isnan`)
Using a GPU tensor in a python `if` statement forces synchronization.
- **Problem**: `if torch.isnan(loss):` forces the CPU to evaluate the boolean condition by pulling the result from the GPU *every single epoch*.
- **Solution**: Move NaN checks to run less frequently (e.g., every 100 epochs alongside logging), or use asynchronous assertion techniques if critical.

### 3. Suboptimal Batch Sizes
Small batch sizes (e.g., $N=1024$) fail to saturate the parallel cores of modern GPUs.
- **Problem**: The overhead of python iteration and launching CUDA kernels dominates the computation. A 3080 12GB can process tens of thousands of points simultaneously.
- **Solution**: For small datasets (e.g., 5000 points), use **Full-Batch Training**. This eliminates the overhead of sampling mini-batches (`torch.randperm`) and minimizes the number of kernel launches per epoch.

## Advanced Optimization Techniques

### 4. PyTorch JIT Compilation (TorchScript / `torch.compile`) — [ATTIVA TRAMITE TORCHSCRIPT]
I PINN richiedono centinaia di lanci di kernel CUDA a causa delle derivate di autograd ad ogni epoca e generano derivate seconde.
- **Evoluzione Ambientale (`torch.compile`)**: Il problema del supporto su Windows e Python 3.14 è stato superato installando `triton-windows`. Tuttavia, `torch.compile` (backend AOTAutograd) presenta un limite architetturale invalicabile con il *double backward* (le derivate seconde spaziali delle PINN generano l'errore `RuntimeError: torch.compile with aot_autograd does not currently support double backward`).
- **Stato e Soluzione**: **TorchScript Attivo**. Per superare il problema senza perdere precisione o riscrivere la matematica in Forward-Mode AutoDiff, viene utilizzato `torch.jit.trace` sui sottomodelli della rete. TorchScript compila il forward pass in un grafo C++ ottimizzato ed è nativamente compatibile con il double backward.
- **Vantaggi**: Riduce l'overhead Python (specie in assenza dei CUDA Graphs) fondendo le operazioni lineari e di attivazione, garantendo un training più veloce in modalità eager.

### 5. Automatic Mixed Precision (AMP) — [SCARTATA / DISATTIVATA]
- **Stato**: **Disattivata di default** (esclusa dalle ottimizzazioni attive).
- **Motivazione**: Il casting a FP16 introdotto da AMP compromette la stabilità dei residui PDE di ordine superiore (come la derivata della derivata del flusso viscoelastic). Provoca instabilità numerica e degrado della precisione scientifica (violando il vincolo di equivalenza matematica). Viene mantenuta disattivata per preservare l'accuratezza.

### 6. CUDA Graphs — [PRESENTE E FUNZIONANTE]
- **Stato**: **Attiva ed equivalente al 100%** per GPU moderne (es. RTX 3080). Porta il throughput da ~2 it/s a **~10.5 it/s** (miglioramento >10x).
- **Hardware-Toggling (GTX 1050 Ti)**: Disattivata automaticamente tramite il controllo centralizzato `IS_1050TI` (VRAM < 4.5 GB) per prevenire crash da Out of Memory (OOM) dovuti all'allocazione di tensori statici.
- **Sfide risolte per l'integrazione**:
  1. *Stream Capture Alignment*: Per evitare l'errore `cudaErrorStreamCaptureImplicit`, sia il *warmup* (3 step fittizi) che la *cattura* del grafo devono avvenire all'interno dello stesso stream CUDA secondario gestito da `CUDAGraphManager`.
  2. *Dynamic Masking*: Rimosse le instanziazioni in-loop di `torch.tensor` in `Viscoelastic_physics.py` (es. `active_mask`) a favore di liste Python e operazioni unrolled che non generano chiamate sincrone host-device.
  3. *Dynamic Weighting & Gradient Fallback*: Poiché `loss.backward()` cancella il grafo di computazione, calcolare i gradienti per la pesatura dinamica (`dynamic_weighting`) o per il logging dei gradienti all'esterno del grafo causava crash. Risolto implementando un **fallback automatico a esecuzione standard** esclusivamente per l'epoca dello step in cui è richiesto il bilanciamento o il log dei gradienti, ripristinando il replay nei passi successivi.

### 7. Forward-Mode AutoDiff — [Pianificata / Futura]
Attualmente, `torch.autograd.grad(..., create_graph=True)` usa il classico *Reverse-Mode AutoDiff*. Nelle PINN abbiamo 2 soli input spaziali ($x, y$) e molti output ($u, v, p, \tau_{xx}, \dots$). Il Reverse-Mode scala linearmente con il numero di output, rendendo il calcolo delle derivate parziali per le PINN estremamente inefficiente.
- **La Soluzione**: Il **Forward-Mode AutoDiff** (es. `torch.autograd.forward_ad`) spinge i gradienti in avanti insieme alla computazione standard della rete. Bastano 2 passate (una per la tangente $x$, una per $y$) per ottenere simultaneamente tutte le derivate incrociate per ogni equazione.
- **Implementazione**: Richiede una riscrittura profonda dei layer fisici per utilizzare il dual tensor system invece del tradizionale backward tracking, ma per PDE complesse è lo "stato dell'arte".

## Related
- **Systems**: [[Viscoelastic_Fluids]]
- **Methods**: [[Loss_History_Tracking]]
