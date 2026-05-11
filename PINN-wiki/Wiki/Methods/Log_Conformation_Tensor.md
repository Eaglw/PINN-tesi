# Log-Conformation Tensor (LCT)

## Overview
Il **Log-Conformation Tensor** è una tecnica di trasformazione variabile utilizzata per risolvere il **High Weissenberg Number Problem (HWNP)** nelle simulazioni numeriche di fluidi viscoelastici. Invece di risolvere direttamente per il tensore degli stress polimerici $\boldsymbol{\tau}$, si risolve per il logaritmo del tensore di conformazione $\mathbf{C}$.

## Motivation
Nei flussi viscoelastici complessi (es. contrazioni, spigoli, flussi attorno a cilindri), lo stress polimerico può subire una crescita **esponenziale** nelle regioni di forte allungamento.
- **Problema nelle PINN**: Le reti neurali faticano a approssimare funzioni con gradienti estremamente ripidi e range di valori che variano di molti ordini di grandezza. Questo porta a instabilità nel training e a errori di convergenza massivi.
- **Soluzione LCT**: Trasformando lo stress in una scala logaritmica, la crescita esponenziale diventa **lineare**, rendendo la superficie della loss molto più liscia e gestibile per l'ottimizzatore Adam/L-BFGS.

## Mathematical Formulation
Il tensore di conformazione $\mathbf{C}$ è legato allo stress polimerico $\boldsymbol{\tau}$ dalla relazione:
$$ \boldsymbol{\tau} = \frac{\mu_p}{\lambda} (\mathbf{C} - \mathbf{I}) $$
Si definisce il tensore logaritmico $\mathbf{S}$:
$$ \mathbf{S} = \log(\mathbf{C}) $$
Nelle equazioni governanti (Oldroyd-B, Giesekus), si sostituisce $\boldsymbol{\tau}$ con l'espressione dipendente da $e^{\mathbf{S}}$. Questo garantisce inoltre che il tensore di conformazione rimanga sempre **definito positivo**, preservando la consistenza fisica del modello.

## Implementation in PINNs (ViscoelasticNet)
Come evidenziato in [[Thakur_et_al_ViscoelasticNet]], l'uso del LCT è essenziale per la scoperta degli stress in regimi di alto Weissenberg ($Wi > 1$). 
- **Output della Rete**: La PINN predice direttamente le componenti di $\mathbf{S}$.
- **Physics Loss**: I residui vengono calcolati trasformando $\mathbf{S}$ in $\boldsymbol{\tau}$ tramite l'esponenziale di matrice prima di entrare nelle equazioni del momento.

## Related
- [[Viscoelastic_Fluids]]
- [[Thakur_et_al_ViscoelasticNet]]
- [[Staged_Precision_Strategy]]
