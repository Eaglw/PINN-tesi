---
title: "Supervisor Meetings: Log & Feedback"
source: "[[04_Supervisor_Meetings_Log.md]]"
author: "Student"
type: "notes"
---

## Summary
Log of meetings with the thesis supervisor, tracking methodological decisions and Q&A sessions regarding PINN implementation for ODEs and PDEs.

## Key Decisions
- **Optimizer Strategy**: Validation of L-BFGS following Adam pretraining.
- **Network Architecture**: Necessity for capacity to scale with problem complexity.
- **Sampling**: Density and position of points significantly impact performance; post-hoc analysis required.
- **Coupled PINN**: Discussions on handling high loss values without data pretraining, normalization of concentration/temperature, and static vs dynamic weight balancing.
- **Inverse Problems**: Utility of working with log values (`ln`) and normalization within the [0, 1] range.

## Related
- [[Staged_Precision_Strategy]]
- [[Dynamic_Weighting]]
- [[Inverse_Problems]]
- [[Nondimensionalization]]
