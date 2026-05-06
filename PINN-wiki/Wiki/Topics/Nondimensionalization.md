# Topic: Nondimensionalization

Nondimensionalization is the process of removing physical units from equations by scaling variables with characteristic values of the system.

## Importance for PINNs
As discussed in [[Hazra_et_al_Convective_Heat_Transfer]], training PINNs on physical units (which can vary by orders of magnitude, e.g., temperature vs. coordinates) often leads to:
- Ill-conditioned Hessian matrices.
- Difficulty in balancing loss terms.
- Slow convergence.

By mapping variables to a dimensionless range (typically \([0, 1]\) or \([-1, 1]\)), the network sees a balanced feature space, which significantly improves training stability.

## Example
In heat transfer, temperature \( T \) is scaled as:
\[ \tilde{T} = \frac{T - T_\infty}{T_0 - T_\infty} \]
where \( T_\infty \) is ambient and \( T_0 \) is initial temperature.

## References
- Core preprocessing step in [[Hazra_et_al_Convective_Heat_Transfer]].
- See [[Heat2D_Analysis]] for coordinate scaling applications.
