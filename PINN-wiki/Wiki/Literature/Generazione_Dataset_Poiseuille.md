# Generazione Dataset Flusso di Oldroyd-B (Poiseuille)

## Summary
Documento di specifica per la generazione di dataset sintetici del flusso viscoelastico di Oldroyd-B stazionario completamente sviluppato (Flusso di Poiseuille) in un canale 2D. Il dataset serve per addestrare e validare Physics-Informed Neural Networks (PINNs) su problemi diretti e inversi.

## Key Methodology
- **Profili Analitici**: Derivazione analitica della velocità $u(y)$ con profilo parabolico governato dalla viscosità totale $\mu_{tot} = \mu_s + \mu_p$.
- **Sforzi Polimerici**: Calcolo analitico degli sforzi viscoelastici:
  - Sforzo di taglio: $\tau_{xy} = \mu_p \frac{du}{dy}$
  - Tensioni normali: $\tau_{xx} = 2 \lambda \mu_p \left(\frac{du}{dy}\right)^2$
  - Componente trasversale nulla: $\tau_{yy} = 0$
- **Funzione di Corrente**: Calcolo analitico di $\psi(y) = \frac{4 u_{max}}{H^2} \left( \frac{H y^2}{2} - \frac{y^3}{3} \right)$ per garantire un campo divergence-free per costruzione.
- **Campionamento**: Supporta sia Grid Sampling (cartesiano regolare) sia Sobol Sampling (quasi-Monte Carlo a bassa discrepanza).
- **Rumore**: Aggiunta opzionale di rumore Gaussiano assoluto o percentuale sui campi generati per simulare dati reali.

## Key Findings & Project Relevance
- Fornisce i profili esatti e consistenti (ground truth) di riferimento per la validazione di modelli come [[ViscoelasticNet]].
- Evidenzia l'importanza del campionamento Sobol per la stabilità e la velocità di convergenza dei modelli PINN.
- Specifica la struttura del dataset salvato in formato `.pt` o `.csv` contenente coordinate ($x, y$), velocità ($u, v$), pressione ($p$), funzione di corrente ($\psi$), e i tensori degli sforzi ($\tau_{xx}, \tau_{xy}, \tau_{yy}$).

## Related Concepts
- **Topics**: [[Viscoelasticity]], [[Fluid_Dynamics]], [[Sampling_Strategies]]
- **Methods**: [[ViscoelasticNet]], [[Log_Conformation_Tensor]]
- **Systems**: [[Viscoelastic_Fluids]]
