## Lesson 1 — Iterations 1-6
**Pattern**: Tapered architecture + SiLU activation + Increasing Epochs (up to 3000).
**Why it worked**: 
- SiLU (Swish) provides smoother gradients for second-order PDEs (Navier-Stokes) compared to Tanh.
- Tapered layers (e.g., [120, 100, 80, 60, 40, 20]) condense fluid features more effectively.
- Adam+LBFGS hybrid needs high epoch counts to escape local minima.
**Metric delta**: -0.1142 (L2 Relative Error: 0.1163 -> 0.0021).

## Lesson 2 — Iterations 7-15
**Pattern**: Extreme Epoch Scaling (4000 to 8000) on fixed Tapered Architecture.
**Why it worked**: 
- The error scaled almost linearly down as epochs increased up to 8000.
- Beyond 8000 (at 10,000), we hit instability/regression.
- The core architecture `[120, 100, 80, 60, 40, 20]` is remarkably robust; attempts to widen it (Iter 8, 11) or deepen it (Iter 15) without scaling points/complexity correctly led to regressions.
**Conditions**: Best results achieved with 1000 collocation points and SiLU.
**Anti-pattern**: 
- Over-widening the network without increasing training duration/points (Iter 8, 11).
- Increasing collocation points (Iter 14) without complementary architecture/epoch adjustments.
- Faster dynamic weight updates (Iter 10) likely caused oscilliations in the loss landscape.
**Metric delta**: -0.0015 (L2 Relative Error: 0.0021 -> 0.0006).

**Current Best Metric**: 0.000646 (Iteration 12)
