# Loss History Tracking

## Overview
Sistema di monitoraggio e visualizzazione della convergenza del modello PINN. Permette di analizzare l'andamento dei diversi componenti della loss (PDE, Data, Boundary) e di identificare eventuali problemi di stabilità numerica o sbilanciamento dei pesi.

## Technical Implementation
- **Logging**: I valori della loss vengono salvati ad ogni epoca (o intervallo di epoche) in un file `history.csv` o gestiti tramite la classe `HistoryTracker`.
- **EMA Smoothing**: Applicazione della Exponential Moving Average per ridurre il rumore nelle curve di loss, facilitando l'identificazione del trend di convergenza a lungo termine. Implementato in [[EMA_Smoothing]].
- **Phase Markers**: Visualizzazione di linee verticali nei plot di loss per demarcare il passaggio tra diverse fasi di training (es. Adam -> L-BFGS o switch tra componenti fisiche nel training viscoelastico).
- **Visualization**: Generazione automatica di plot logaritmici che confrontano i contributi relativi di ogni termine della funzione obiettivo.

## Application
Fondamentale per il debugging di modelli complessi come [[ViscoelasticNet]], dove il monitoraggio separato dei residui di continuità, quantità di moto e costitutivi è essenziale per il successo del training.

## References
- [[EMA_Smoothing]]
- [[Viscoelastic_Metrics]]
- [[Viscoelastic_Plotting_Updates]]
