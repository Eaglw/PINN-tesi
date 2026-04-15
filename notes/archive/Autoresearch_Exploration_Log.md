# Autoresearch Exploration Log: Heat2D Optimization

## 📅 Status as of 2026-04-01 (Session Continuation)

### 🎯 Objective
Minimize **L2 Relative Error** in the 2D Heat Transfer problem using PINNs. 
Baseline reached: **0.0400** (80x6 GELU, PurePhys, 4000 Adam + 5000 L-BFGS).

### 📊 Analysis of Previous Iterations
- **Architecture**: Deep and narrow (80x6) performed better than wide and shallow (150x3). Deepest (100x8) didn't provide a massive jump over 80x6, suggesting a plateau in capacity or training efficiency.
- **Activation**: GELU is the clear winner over Tanh/SiLU.
- **Run Type**: PurePhys is sufficient and cleaner than DataPhys for this problem.
- **Weighting**: `update_weights_every=100` is the most stable.
- **Collocation**: Increasing points (60x60) actually *increased* error slightly in a 4000-epoch run, likely due to needing more epochs to resolve the higher-frequency information or better balancing.

---

## 🔬 Phase 2: Hypotheses & New Experiments

### Hypothesis 1: Tapered Architecture
*Maybe a funnel-like structure allows the network to compress information better.*
- **Plan**: Test `[120, 100, 80, 60, 40, 20]` vs the flat `80x6`.

### Hypothesis 2: Weighted BC focus
*Boundary conditions are the "anchor" of this problem. If the BC error is high, the interior cannot be correct.*
- **Plan**: Start with higher BC weight (`bc: 10.0, physics: 1.0`) and let dynamic weighting adjust.

### Hypothesis 3: Refined Collocation Sampling
*The current grid might be missing gradients near the corners.*
- **Plan**: Use a non-uniform grid or increase points specifically near boundaries. (Will start with non-uniform grid logic if possible).

---

## 🏃 Run Log

### Iteration 11: Tapered Architecture
- **Config**: `--arch 120,100,80,60,40,20 --act GELU --epochs 2000 --lbfgs_iter 500`
- **Result**: **0.060375** (Better than 80x6 which was 0.0647 in 2000 epochs).

### Iteration 12: Initial BC weighting
- **Hypothesis**: Giving more weight to BC at the start will anchor the solution.
- **Config**: `--arch 80,80,80,80,80,80 --bc_weight 10.0`
- **Result**: **0.054747** (Better than baseline 0.0647). Promising.

### Iteration 13: Combined Tapered + BC weighting
- **Hypothesis**: Synergy between architecture capacity and BC anchoring.
- **Config**: `--arch 120,100,80,60,40,20 --bc_weight 10.0`
- **Result**: **0.044160** (Excellent jump! Almost reached the 4000-epoch baseline in half the time).

### Iteration 14: Aggressive BC weighting
- **Hypothesis**: Can we push it further with bc_weight=50?
- **Config**: `--arch 120,100,80,60,40,20 --bc_weight 50.0`
- **Result**: **0.039440** (BREAKTHROUGH! Beat the 4000-epoch baseline of 0.040 in just 2000 epochs).

### Iteration 15: Wider Tapered Arch
- **Hypothesis**: Maybe more capacity in the first layers helps.
- **Config**: `--arch 200,150,100,50 --bc_weight 10.0`
- **Result**: **0.050295** (Good, but depth/tapering seems more important than width at the start).

### Iteration 16: Deeper Tapered + Aggressive BC
- **Hypothesis**: Combine depth with the winning BC weight.
- **Config**: `--arch 150,120,100,80,60,40,20 --bc_weight 50.0`
- **Result**: **0.041780** (Slightly worse than Iteration 14, suggesting 120-20 is the optimal depth/taper).

### 🏆 Final Optimization: Scale-up
- **Config**: `--arch 120,100,80,60,40,20 --bc_weight 50.0 --epochs 20000 --lbfgs_iter 10000`
- **Result**: **0.038466** (Best L2 error recorded so far).

## 🚀 Phase 5: Autonomous Optimization (Autoresearch Round 1)

### 🎯 Objective: Automated Hyperparameter & Structural Search
Utilize the `autoresearch` framework to test hypotheses in rapid succession.

### 🏃 Phase 5 Run Log (Iterations 1-20)

| Iter | Hypothesis | Config | L2 Error | Status |
|---|---|---|---|---|
| 1 | Deeper Tapered Arch | [140, 120, 100, 80, 60, 40, 30, 20] | **0.028684** | **KEEP** |
| 2 | Neuron-wise LAA | Separate 'a' for each neuron | 0.032478 | DISCARD |
| 3 | Pure Sobol Sampling | 1600 Sobol points (replacing Grid) | **0.020703** | **KEEP** |
| 4 | High Density Collocation | 60x60 (3600 points) | 0.033689 | DISCARD |
| 5 | Steeper LAA Init | Initialize 'a' at 1.2 | 0.039568 | DISCARD |
| 6 | Higher BC Weight | bc_weight = 100 | 0.040464 | DISCARD |
| 7 | SiLU Activation | Switch GELU to SiLU | 0.021955 | DISCARD |
| 8 | Wider Initial Layer | [200, 140, 120...20] | 0.028297 | DISCARD |
| 9 | Step LR Scheduler | Decay every 1000 epochs | 0.024090 | DISCARD |
| 10 | Scale-up (10k epochs) | Best config on 10k epochs (Step LR) | 0.025115 | DISCARD |
| 11 | Scale-up (10k epochs) | Best config on 10k epochs (Plateau) | 0.021360 | DISCARD |
| 12 | **High Density BC** | **num_b_side = 100 (400 total)** | **0.019943** | **KEEP** |
| 13 | Hybrid Sampling | Grid 20x20 + Sobol 1200 | 0.021050 | DISCARD |
| 14 | Even Deeper Arch | [140...50...20] (9 layers) | 0.024925 | DISCARD |
| 15 | Normal LAA Init | a ~ N(1.0, 0.05) | 0.021540 | DISCARD |
| 16 | **Extended L-BFGS** | **lbfgs_iter = 2000** | **0.018115** | **KEEP** |
| 18 | Med Density Collocation | 50x50 (2500 points) | 0.018385 | DISCARD |
| 19 | Adam Warmup | 500 epochs at 1e-4 | 0.022848 | DISCARD |
| 20 | Very Wide Tapered | [160, 140...20] | 0.026210 | DISCARD |

### 🏆 Phase 5 Winner: Iteration 16
- **L2 Relative Error: 0.018115**
- **Config**: Tapered [140-20], GELU LAA (layer-wise), Pure Sobol (1600), BC Density (400 pts), BC Weight 50, Adam 5000 + L-BFGS 2000.

---

## 🚀 Phase 6: Precision Refinement (Autoresearch Round 2)

### 🎯 Objective: Breakthrough to L2 < 0.0150
Focus on coordinate normalization, regularization, and specialized sampling.

### 🔬 Hypotheses to Explore (Iter 21-30)

#### Hypothesis 9: Coordinate Scaling to [-1, 1]
*Scaling input domain to [-1, 1] often improves gradient flow and prevents saturation in GELU/Tanh.*
- **Plan**: Modify `Heat2D_adaptive_mini.py` to map [0, 1] -> [-1, 1].

#### Hypothesis 10: L2 Regularization (Weight Decay)
*Small weight decay might prevent the LAA parameters from drifting too far in long runs.*
- **Plan**: Add `weight_decay=1e-6` to Adam.

#### Hypothesis 11: Gradient-based Sampling (SAR Hybrid)
*Combine Sobol with points added where the gradient of the solution is highest.*
- **Plan**: Target corners and boundaries specifically.

#### Hypothesis 12: Sine/Cosine Positional Encoding (RFF)
*Random Fourier Features or Positional Encoding could help resolve high-frequency components.*
- **Plan**: Map inputs $(x, y) \to (\sin(kx), \cos(kx), \sin(ky), \cos(ky))$.

---

## 🏃 Phase 6 Run Log (Iterations 21-30)

| Iter | Hypothesis | Config | L2 Error | Status |
|---|---|---|---|---|
| 21 | **Scale coordinates to [-1, 1]** | Maps [0, 1] -> [-1, 1] | **0.009847** | **KEEP** |
| 22 | L2 Regularization | weight_decay = 1e-6 | 0.009847 | DISCARD |
| 23 | Multi-phase SAR | Add high-res points every 1000 | 0.016204 | DISCARD |
| 24 | Uniform Init for 'a' | a ~ U[0.9, 1.1] | 0.012330 | DISCARD |
| 25 | Cosine Annealing LR | Switch from Plateau | 0.010714 | DISCARD |
| 26 | Wider initial arch | [180, 140...20] | 0.013399 | DISCARD |
| 27 | Single-phase SAR | Add 400 pts at midpoint | 0.012436 | DISCARD |
| 28 | Higher Sobol Density | 2000 points | 0.010478 | DISCARD |
| 29 | Init a=0.9 | Lower initial slope | 0.740395 | DISCARD (bug) |
| 30 | Init a=1.1 | Higher initial slope | 0.011764 | DISCARD |

### 🏆 Phase 6 Winner: Iteration 21
- **L2 Relative Error: 0.009847**
- **Config**: Tapered [140-20], GELU LAA (layer-wise, a=1.0), Pure Sobol (1600), BC Density (400 pts), BC Weight 50, Domain [-1, 1], Adam 5000 + L-BFGS 2000.

## 🏁 Final Conclusions
1. **Coordinate Scaling**: Mapping inputs to `[-1, 1]` had the most profound impact, bringing error down from ~0.018 to under 0.01.
2. **Architecture**: A deep tapered structure `[140, 120, 100, 80, 60, 40, 30, 20]` provides the ideal capacity funnel.
3. **Sampling**: 1600 Sobol points without dynamic refinement outperformed all adaptive methods (SAR) which disrupted Adam's momentum.
4. **Overall Progress**: From baseline `0.0400` to `0.0098` (75% error reduction) via pure structural and algorithmic optimization without increasing compute time excessively.
