---
name: organize_run
description: Protocollo post-run per la categorizzazione, rinominazione arricchita delle cartelle di output ed estrazione dettagliata delle metriche in SUMMARY_RUNS.md e PINN-Wiki.
---

# organize_run: Protocollo di Post-Elaborazione Run & Curation Output

La skill `organize_run` trasforma Antigravity in un gestore e catalogatore automatico dei risultati numerici del progetto PINN-tesi. Deve essere invocata al termine di ogni esperimento (es. *"Antigravity, esegui organize_run"* o *"organizza l'ultima run"*).

---

## 1. Obiettivi della Skill

1. **Categorizzazione & Rinominazione Cartelle**:
   - Rinominare la cartella dell'ultima run (o della cartella specificata) in `final_roll/output_4rollmill/` anteponendo:
     - **Categoria**: `[DIRETTO]`, `[INVERSO]`, `[SOLO_PRESSIONE]`, `[CRASHED_...]`, `[INCOMPLETE_...]`.
     - **Flag di Errore L2**: es. `[ErrU_3.2e-3]`.
     - **Flag Epoche & Dettagli**: es. `[151k_epochs]`, `[5plots]`, `[allTBC]`, `[LambdaAdam]`.
     - **Recap del tentativo**: sintesi delle modifiche rispetto al codice precedente.

2. **Aggiornamento Dettagliato di `SUMMARY_RUNS.md`**:
   - Estrarre tutte le metriche da `train_log.txt` (errore $L_2(u,v)$, $L_2(p)$, $L_2(\tau)$, loss finale, GPU VRAM usata) ed inserire/aggiornare la voce nel file `final_roll/output_4rollmill/SUMMARY_RUNS.md`.

3. **Sincronizzazione Obsidian Wiki**:
   - Creare o aggiornare la scheda markdown della run in `PINN-wiki/Runs/` mantenendo allineato il diario di bordo.

4. **Gestione Pulizia Run Fallite**:
   - Se la run è fallita per eccezioni (`KeyboardInterrupt`, `NameError`, `OOM`) o senza log, proporne l'eliminazione o marcarla con `[CRASHED_...]`.

---

## 2. Flusso Operativo Passo-Passo

### Step 1: Ispezione & Parsing dell'Output Target
- Identificare la cartella di output più recente in `final_roll/output_4rollmill/` (o quella indicata dall'utente).
- Eseguire lo script di parsing helper:
  ```powershell
  .\venv\Scripts\python .agents/skills/organize_run/scripts/organize_run.py
  ```
- Leggere il file `train_log.txt` e la presenza di `checkpoint.pth` / grafici `.png`.

### Step 2: Estrazione delle Metriche
Dati da estrarre dal log:
- **Status**: Completed / Interrupted / Crashed.
- **Tipo Esperimento**: Direct Problem, Inverse Problem (stima $\lambda$/$Wi$), Solo Pressione, MLS derivatives.
- **Errore L2 Velocità**: $L_2(u), L_2(v)$.
- **Errore L2 Pressione**: $L_2(p)$.
- **Errore L2 Stress**: $L_2(\tau_{xx}), L_2(\tau_{xy}), L_2(\tau_{yy})$.
- **Epoche**: Numero totale di iterazioni Adam e L-BFGS.

### Step 3: Rinominazione Cartella
Applicare la convenzione standard dei nomi:
`[CATEGORIA]_[ErrU_VALORE]_[METRICHE_ED_EPOCHE]_[RECAP_TENTATIVO]_YYYYMMDD_HHMMSS`

Esempi:
- `[DIRETTO]_[ErrU_3.2e-3]_[BEST_ACCURACY]_[151k_epochs]_20260709_162220`
- `[INVERSO]_[ErrU_1.8e-3]_[5plots]_[allTBC_Run003]_20260720_191853`
- `[CRASHED_NameError]_[30k_epochs]_20260725_142022`

### Step 4: Aggiornamento `SUMMARY_RUNS.md`
Aggiornare la tabella markdown in `final_roll/output_4rollmill/SUMMARY_RUNS.md` includendo:
- Nome cartella arricchito
- Categoria
- Errore L2 $u,v$ e $p$
- Presenza grafici e checkpoint
- Note dettagliate sulle modifiche introdotte nella run

### Step 5: Report Finale all'Utente
Fornire un breve resoconto all'utente mostrando:
- Vecchio nome ➔ Nuovo nome della cartella
- Sintesi delle metriche raggiunte
- Link diretto a `SUMMARY_RUNS.md`
