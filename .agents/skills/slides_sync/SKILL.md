---
name: slides_sync
description: Sincronizza, scarica, legge ed estrae testo, note del relatore e differenze delle slide della presentazione di tesi da Google Slides o dai file PPTX/PDF locali.
---

# slides_sync: Gestione & Sincronizzazione Slide Seminario Tesi PINN

Questa skill gestisce l'accesso diretto, lo scaricamento automatico, l'estrazione testuale, le note del relatore (*speaker notes*) e il versionamento (*diff*) delle slide per la tesi di laurea:
**"Characterization of viscoelastic flows in a four-roll mill with physics-informed neural networks"** (Seminari I e II).

---

## 1. Riferimenti Presentazione Google Slides

- **ID Documento Predefinito**: `1bTiAcKGMXmr67rRc2oLwJqCIH163FC8jySCspP88fcs`
- **Link Modifica**: `https://docs.google.com/presentation/d/1bTiAcKGMXmr67rRc2oLwJqCIH163FC8jySCspP88fcs/edit?usp=sharing`
- **Export Diretto PPTX**: `https://docs.google.com/presentation/d/1bTiAcKGMXmr67rRc2oLwJqCIH163FC8jySCspP88fcs/export/pptx`
- **Export Diretto PDF**: `https://docs.google.com/presentation/d/1bTiAcKGMXmr67rRc2oLwJqCIH163FC8jySCspP88fcs/export/pdf`

---

## 2. File Locali Gestiti

- **File Master Ufficiali**:
  - `final_roll/presentation_assets/Seminario I e II - Tesi PINN.pptx`
  - `final_roll/presentation_assets/Seminario I e II - Tesi PINN.pdf`
- **Estratti & Dump di Consultazione**:
  - `scratch/slides_dump.txt`: baseline testuale completo delle slide (per diff rapido).
  - `scratch/clean_speaker_notes.txt`: note del relatore, battute discorsive e promemoria di regia per ciascuna slide.
  - `scratch/final_slides_and_notes.txt`: dump combinato di testi a schermo e note.

---

## 3. Comandi di Esecuzione Rapida

Per sincronizzare ed estrarre le slide aggiornate in un unico passaggio:

```powershell
# Da root del repository:
.\venv\Scripts\python .agents/skills/slides_sync/scripts/sync_slides.py

# Se invocato dall'interno di final_roll o PINN-wiki:
..\venv\Scripts\python ../.agents/skills/slides_sync/scripts/sync_slides.py
```

Lo script effettua automaticamente:
1. Download della versione più recente in formato `.pptx` e `.pdf` salvandola in `final_roll/presentation_assets/`;
2. Parsing XML delle slide e delle note associate (`notesSlide`);
3. Confronto con il dump precedente ed evidenziazione a console delle slide modificate;
4. Aggiornamento dei dump in `scratch/`.

---

## 4. Linee Guida per l'Agente

Quando l'utente richiede frasi come:
- *"leggi le slide"*
- *"aggiorna le slide da Google Slides"*
- *"cosa è cambiato nella presentazione"*
- *"mostrami le note della slide N"*

L'agente deve:
1. Eseguire `sync_slides.py` con `run_command` per garantire che i file locali riflettano l'ultimo stato online.
2. Leggere `clean_speaker_notes.txt` o `final_slides_and_notes.txt` per rispondere con precisione ai quesiti specifici.
3. Indicare chiaramente eventuali discrepanze rispetto al baseline.
