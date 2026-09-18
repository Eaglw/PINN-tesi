---
name: daily_recap
description: Analyzes all agent transcripts and sessions of the current day, aggregates git changes and benchmark metrics, and generates a comprehensive technical daily log in the PINN-tesi Obsidian Wiki.
---

# daily_recap: Autonomous Daily Project Intelligence & Wiki Logger

## 1. Skill Overview & Purpose
The `daily_recap` skill autonomously extracts, analyzes, and synthesizes everything that happened across all Antigravity agent sessions during a given day. It bridges the gap between active pair programming and long-term project memory by producing a high-impact, pedagogical, and rigorous "Giornale di Bordo Tecnico" saved directly into the **PINN-tesi Obsidian Wiki**.

It does NOT produce a shallow bullet-point list or a raw transcript dump: it synthesizes the day's work across fluid physics, numerical CFD benchmarks, neural network architecture, mathematical proofs/metrics, and operational roadmaps.

---

## 2. Invocation Triggers
This skill is triggered when the user asks to:
- *"Esegui daily_recap"*
- *"Fai il punto della giornata e salvalo nella wiki"*
- *"Aggiorna la wiki a fine giornata con quanto fatto oggi"*
- *"Fai un recap completo di tutte le chat di oggi e traccia l'avanzamento"*

---

## 3. Step-by-Step Execution Workflow

### Step 1: Automated Transcript & Git Extraction
Run the dedicated extraction script from the repository root:
```powershell
.\venv\Scripts\python .agents/skills/daily_recap/scripts/extract_daily_digest.py
```
*(Optionally pass `--date YYYY-MM-DD` if reviewing a past date).*

The script automatically:
1. Scans `~/.gemini/antigravity/brain` for all conversation sessions modified on the target date.
2. Extracts chronological user prompts, file modifications (`write_to_file`, `replace_file_content`), and critical shell commands.
3. Queries `git log` and `git diff --stat` for exact repository modifications.
4. Aggregates any `metrics_summary.json` generated in `final_roll/output_4rollmill/`.

### Step 2: Knowledge Synthesis (The 7 Canonical Sections)
Structure the daily report using the following mandatory 7-section framework:

1. **Sintesi Esecutiva & Macro-Risultati**: High-level overview of breakthroughs, bugs solved, or major decisions made during the day.
2. **Avanzamento Numerico & Simulazioni (COMSOL & PINN)**:
   - Results of mesh convergence, grid independence benchmarks, GCI, error tables ($L_2$, $L_1$, $L_\infty$).
   - Explicit distinction between FEM discretization convergence and PINN representation/optimization dynamics (**La Dualità di Convergenza**).
3. **Fisica del Problema & Fluidodinamica**:
   - Physical rationale behind observed behaviors (e.g., why $\tau_{xy} \equiv 0$ on axes, why normal stresses form "W" and "M" profiles due to incompressibility $\nabla \cdot \mathbf{u}=0$, convective memory between inflow and outflow).
4. **Fondamenti Matematici per Tesi ed Esame**:
   - Theoretical explanations of key metrics or formulas (e.g., origin and derivation of Relative $L_2$ error in Hilbert space $L^2(\Omega)$, comparison with MSE, $L_1$, $L_\infty$).
5. **Evoluzione Architetturale & Codice**:
   - Code changes, new trainable parameters, regularization (e.g., sigmoid/softplus bounds), learning rate schedules, batch runners, VRAM management.
6. **Roadmap Operativa & Timeline Immediata**:
   - Multi-device delegation (PC Mauri vs Local PC), pending overnight/morning runs, sequential batch execution, next sweep steps.
7. **Materiale per Presentazioni / Seminari / Tesi**:
   - Visual ideas (e.g., multi-mesh PINN cutline overlay), storyline structure slide-by-slide for PowerPoint or thesis chapters.

### Step 3: File Placement in the Wiki
1. Create or overwrite the daily log page:
   - Path: `PINN-wiki/Wiki/Daily_Logs/YYYY-MM-DD.md`
2. Update the master log:
   - Prepend/insert an entry in `PINN-wiki/Wiki/01_Log.md` referencing `[[YYYY-MM-DD]]` with a 3-bullet summary.
3. Update the vault index:
   - In `PINN-wiki/Wiki/00_Index.md`, ensure the `Daily Logs` section contains a link to `[[YYYY-MM-DD]]`.

### Step 4: Link Integrity & Self-Healing
Execute the link integrity check to ensure zero broken links across the Obsidian vault:
```powershell
python -c "import re, glob, os; links = set(re.findall(r'\[\[([^\]\|#]+)', open('PINN-wiki/Wiki/Daily_Logs/YYYY-MM-DD.md', encoding='utf-8').read())); pages = {os.path.splitext(os.path.basename(p))[0] for p in glob.glob('PINN-wiki/Wiki/**/*.md', recursive=True)}; missing = links - pages - {''}; print('Missing wikilinks:', missing) if missing else print('All Wiki links OK!')"
```
If any atomic page is missing, create it or fix the reference immediately.

---

## 4. Daily Log Markdown Template

```markdown
# Daily Log: YYYY-MM-DD — [Titolo Sintetico della Giornata]

- **Data**: YYYY-MM-DD
- **Sessioni Antigravity**: N sessioni
- **Focus Principale**: [Breve frase riassuntiva]
- **Wiki Tag**: [[Daily_Logs]], [[00_Index]], [[01_Log]]

---

## 1. Sintesi Esecutiva & Macro-Risultati

...

## 2. Avanzamento Numerico & Simulazioni (COMSOL vs PINN)

...

## 3. Approfondimento Fisico: Fluidodinamica e Tensori di Sforzo

...

## 4. Basi Matematiche & Metriche (Focus Esame/Seminario)

...

## 5. Modifiche al Codice & Architettura PINN

...

## 6. Roadmap Operativa & Setup Multi-Device

...

## 7. Spunti per Seminario, Slide e Tesi

...

---
## Riferimenti & Wikilinks
- [[Mesh_Convergence_Protocol]]
- [[Viscoelastic_Training]]
- [[ViscoelasticNet_Full model]]
```
