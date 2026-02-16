# Implementation Plan - LR Change Visualization

Implement tracking and visualization of learning rate changes in training loss plots for the Heat2D problem across all relevant source files and main scripts.

## Phase 1: Core Logic Updates
- [ ] Task: Update `TrainingHistory` in `func/history_tracker.py`
    - [ ] Modify `update()` to accept an optional `lr` parameter (o estrarlo dal `loss_dict`) e memorizzarlo in una lista dedicata.
    - [ ] Aggiornare `plot_losses()` per identificare le epoche in cui il LR cambia valore.
    - [ ] Aggiungere linee verticali tratteggiate (`axvline`) per ogni cambio di LR rilevato.
- [x] Task: Conductor - User Manual Verification 'Core Logic' (Protocol in workflow.md)

## Phase 2: Integration in Heat2D Source Modules
- [ ] Task: Update `Heat2D/src/Heat2D_NN.py`
    - [ ] Estrarre il LR corrente e passarlo a `loss_history.update()`.
- [ ] Task: Update `Heat2D/src/Heat2D_NN_griglia.py`
    - [ ] Estrarre il LR corrente e passarlo a `loss_history.update()`.
- [ ] Task: Update `Heat2D/src/Heat2D_PINN.py`
    - [ ] Estrarre il LR corrente e passarlo a `loss_history.update()`.
- [x] Task: Conductor - User Manual Verification 'Source Modules' (Protocol in workflow.md)

## Phase 3: Integration in Inverse Problem
- [ ] Task: Update `Heat2D/Heat2D_inverse_main.py`
    - [ ] In `run_inverse_experiment`, estrarre il LR dell'ottimizzatore ad ogni epoca (sia Adam che L-BFGS se pertinente) e passarlo a `history.update()`.
- [ ] Task: Conductor - User Manual Verification 'Inverse Main' (Protocol in workflow.md)

## Phase 4: Finalization & Verification
- [ ] Task: Verify Main Scripts Coverage
    - [ ] Confermare che `Heat2D_main.py`, `Heat2D_reduced_main.py` e `Heat2D_weighted_main.py` siano coperti tramite le modifiche ai moduli `src`.
- [ ] Task: Manual Verification by User
    - [ ] L'utente eseguirà manualmente i test per verificare la comparsa delle linee verticali nei grafici.
- [ ] Task: Conductor - User Manual Verification 'Finalization' (Protocol in workflow.md)
