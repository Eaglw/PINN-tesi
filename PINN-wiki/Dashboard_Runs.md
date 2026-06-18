---
tags: [dashboard, runs]
---

# 🚀 Runs Dashboard

Questa dashboard raccoglie automaticamente tutti i file di log generati dai training (presenti nella cartella `Runs/`). Le run sono ordinate per data (dalla più recente).

> [!NOTE] 
> Questa dashboard è pensata per essere utilizzata con **Obsidian Bases** (o plugin simili come Database Folder / Projects). 
> Puoi creare o configurare qui la vista tabellare che punterà alla cartella `Runs/`, in cui ogni Run è archiviata e indicizzata dalle proprie Properties (YAML frontmatter).

## Struttura della Cartella `Runs/`
Per ogni training completato, troverai una cartella denominata `Run_XXX_ConfigName`. All'interno vi è:
- Un file Markdown (la nota vera e propria indicizzata qui sopra)
- Il log testuale del training (`train_log.txt`)
- Le varie immagini generate (`loss_history.png`, grafici del campo, ecc.)
