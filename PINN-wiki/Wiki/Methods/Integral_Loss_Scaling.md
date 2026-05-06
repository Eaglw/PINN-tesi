# Method: Integral Loss Scaling

Integral scaling is a methodology where loss terms are scaled proportionally to the volume, area, or length of the domain they represent.

## Rationale
In discrete loss formulations, a large number of internal collocation points can dominate the training loss, causing the optimizer to ignore boundary conditions. Integral scaling ensures that $L_{PDE}$ and $L_{bc}$ contribute proportionally to the total objective.

## Implementation
Each loss term is multiplied by the measure (volume/area) of its respective domain:
$$ L_{total} = k_{PDE} \int_\Omega |f|^2 d\Omega + k_{BC} \int_{\partial\Omega} |g|^2 ds $$

## References
- Discussed as an advanced tool in [[Sharma_et_al_Hyperparameter_Selection]].
