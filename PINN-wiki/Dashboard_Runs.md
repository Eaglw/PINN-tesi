---
tags: [dashboard, runs]
---

# 🚀 Runs Dashboard

Questa dashboard raccoglie automaticamente tutti i file di log generati dai training (presenti nella cartella `Runs/`). Le run sono ordinate per data (dalla più recente).

> [!NOTE] 
> Questa tabella è generata tramite il plugin **Dataview**. Assicurati di averlo installato e attivato.

```dataview
TABLE 
    status AS "Status", 
    type AS "Tipologia", 
    inverse_problem AS "Inverso?", 
    dataset AS "Dataset",
    epochs AS "Epoche"
FROM "Runs"
SORT date DESC
```

## Struttura della Cartella `Runs/`
Per ogni training completato, troverai una cartella denominata `Run_XXX_ConfigName`. All'interno vi è:
- Un file Markdown (la nota vera e propria indicizzata qui sopra)
- Il log testuale del training (`train_log.txt`)
- Le varie immagini generate (`loss_history.png`, grafici del campo, ecc.)
