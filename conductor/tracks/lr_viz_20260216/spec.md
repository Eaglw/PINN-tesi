# Spec: LR Change Visualization in Loss Plots

## Overview
Implementare una funzionalità di logging e visualizzazione per tracciare i cambiamenti del Learning Rate (LR) durante il training. Il sistema deve indicare graficamente le epoche in cui il LR è stato modificato (es. dimezzato) tramite linee verticali sottili nei grafici della loss.

## Functional Requirements
- **Logging**: La classe `TrainingHistory` deve poter memorizzare il valore del LR per ogni epoca.
- **Detection**: Durante la generazione del grafico (`plot_losses`), la classe deve identificare automaticamente le epoche in cui il valore del LR è cambiato rispetto all'epoca precedente.
- **Visualization**:
    - Aggiungere linee verticali (es. `axvline`) in corrispondenza delle epoche di cambio LR.
    - Le linee devono essere sottili e distinguibili (es. colore grigio o tratteggiate) per non coprire le curve della loss.
    - Funzionamento garantito sia per `StepLR` che per `ReduceLROnPlateau`.
- **Scope**: L'implementazione deve essere integrata nei file core di `Heat2D` (NN e PINN) e in `Heat2D_inverse_main.py`.

## Technical Changes
- **`func/history_tracker.py`**:
    - Aggiornare `TrainingHistory.__init__` per includere `self.lr_history`.
    - Aggiornare `update()` per accettare e memorizzare il valore del LR.
    - Aggiornare `plot_losses()` per calcolare i punti di discontinuità nel LR e disegnare le linee verticali.
- **`Heat2D/src/Heat2D_NN.py`**, **`Heat2D/src/Heat2D_NN_griglia.py`** & **`Heat2D/src/Heat2D_PINN.py`**:
    - Estrarre il LR corrente dall'ottimizzatore/scheduler ad ogni epoca e passarlo a `history.update()`.
- **`Heat2D/Heat2D_inverse_main.py`**:
    - In `run_inverse_experiment`, estrarre il LR dell'ottimizzatore ad ogni epoca e passarlo a `history.update()`.

## Acceptance Criteria
- Il grafico `NNloss_history.png` e `PINNloss_history.png` devono mostrare linee verticali nelle epoche in cui il LR è cambiato.
- La funzionalità deve funzionare correttamente sia nella fase Adam che (se applicabile) nel rilevamento iniziale.
- Nessuna regressione nelle prestazioni di training.
