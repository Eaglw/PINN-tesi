# Thesis Writing Tracker & Roadmap (10 Giorni - 2h/giorno)

**Obiettivo:** Finalizzazione della tesi magistrale in Ingegneria Chimica (*Physics-Informed Neural Networks per fluidi viscoelastici nel Four-Roll Mill*).  
**Formato sessioni:** Blocchi operativi mirati da max 2 ore al giorno (task modulari da 30-45 min).  
**Ultimo aggiornamento:** 2026-09-16  
**Stato Attuale:** **Giorno 1 (G1) - In corso**

---

## Roadmap e Stato Avanzamento

- [ ] **G1: Ristrutturazione Cap 2 (Fluidodinamica)**
  - File: `Chapters/2.Fluid-dynamics.tex`
  - Output: Indice dettagliato con raccordo al Four-Roll Mill. Taglio del superfluo/generico dalla bozza.
  - Stato: In corso.
- [ ] **G2: Revisione testo Cap 2 (Parte 1)**
  - File: `Chapters/2.Fluid-dynamics.tex`
  - Output: Sezioni 2.1 e 2.2 chiuse (Governing equations, Modelli costitutivi: focus su limiti sperimentali e necessità numerica).
- [ ] **G3: Chiusura Cap 2 (Parte 2)**
  - File: `Chapters/2.Fluid-dynamics.tex`
  - Output: Sezioni 2.3 e 2.4 chiuse (Adimensionalizzazione specifica del dominio, cinematica del four-roll mill).
- [ ] **G4: Ristrutturazione e revisione Cap 3 (PINN - Parte 1)**
  - File: `Chapters/3.PINNs.tex`
  - Output: Fondamenti di Raissi et al., struttura loss PDE + BC/IC, formulazione stream-function.
- [ ] **G5: Chiusura Cap 3 (PINN - Parte 2)**
  - File: `Chapters/3.PINNs.tex`
  - Output: Stato dell'arte su reti per fluidi complessi/viscoelastici (High-Weissenberg problem, gradient pathologies, staged training).
- [ ] **G6: Cap 4: Metodologia e Setup**
  - File: `Chapters/4.Results.tex` (da strutturare/creare)
  - Output: Dominio 4-roll mill, campionamento punti di collocazione, architettura multi-head, loss functions implementate.
- [ ] **G7: Cap 4: Validazione COMSOL (Fase 1)**
  - File: `Chapters/4.Results.tex`
  - Output: Benchmark sintetico COMSOL, metriche di errore e confronto Fase 1 (cinematica e reologia).
- [ ] **G8: Cap 1: Introduzione e Motivazione**
  - File: `Chapters/1.Introduction.tex` (da strutturare/creare)
  - Output: Stesura completa: traiettoria da "limiti reometria classica" a "PINN nel four-roll mill".
- [ ] **G9: Cap 4: Risultati Fase 2 o Congelamento Stato**
  - File: `Chapters/4.Results.tex`
  - Output: Identificazione parametri / viscosità, analisi critica della convergenza e momentum coupling.
- [ ] **G10: Conclusioni + Raccordo Finale**
  - File: `Chapters/5.Conclusions.tex` & `references.bib`
  - Output: Conclusioni e future work, audit bibliografico completo, verifica coerenza notazioni globali.

---

## Convenzioni di Notazione e Raccordo Fisico-PINN

| Grandezza / Concetto | Notazione LaTeX | Codice PINN / Corrispondenza | Note di Coerenza |
| :--- | :--- | :--- | :--- |
| Funzione di corrente | $\psi$ | `model_psi` | $u = \partial\psi/\partial y$, $v = -\partial\psi/\partial x$ (incompressibilità intrinseca) |
| Pressione | $p$ | `model_p` | Fase 2 attiva; mai congelare $\psi$ in Fase 2 |
| Extra-stress tensor | $\boldsymbol{\tau}$ ($\tau_{xx}, \tau_{xy}, \tau_{yy}$) | `model_tau` | Tensore simmetrico 2D; Oldroyd-B constitutive equation |
| Numero di Weissenberg | $\text{Wi} = \frac{\lambda U}{L}$ | `Wi` | Parametro di rilassamento |
| Numero di Reynolds | $\text{Re} = \frac{\rho U L}{\mu_{tot}}$ | `Re` | Flusso strisciante / inerzia trascurabile se $\text{Re} \ll 1$ |
| Rapporto di viscosità | $\beta = \frac{\mu_s}{\mu_{tot}}$ | `beta` | $\mu_{tot} = \mu_s + \mu_p$; split solvente/polimero |
| Tag parametrico | `L{lambda}-P{eta_p}-S{eta_s}` | Es: `L0.05-P0.5-S0.5` | Standard tassativo per dataset, checkpoint e run |

---

### Sessione G1 - 2026-09-16
- **Obiettivo:** Inizializzazione tracker e ristrutturazione Capitolo 2 (*Viscoelastic Fluid Mechanics*).
- **Azioni completate:**
  - Creato framework di persistenza e `THESIS_TRACKER.md`.
  - Configurate regole operative in `GEMINI.md`.
  - **Capitolo 2 - Intro:** Riformulata con taglio ingegneristico verso il formalismo differenziale della PINN.
  - **Capitolo 2 - Sez 2.1:** Rimossa l'incomprimibilità prematura da 2.1.1 (puramente generale).
  - **Capitolo 2 - Sez 2.2:** Approfondito il campo solenoidale (senza stream function, rimandata a Cap 4); corretta la gravità trascurabile (eliminato $Bo$, sostituito con rapporto peso/sforzi viscosi $\rho g L^2 / (\mu_0 U) \ll 1$); giustificata planarità 2D e stazionarietà.
  - **Capitolo 2 - Sez 2.3:** Eliminati completamente PTT e Giesekus; focalizzazione esclusiva su Oldroyd-B e spiegazione del comportamento estensionale singolare / High-Weissenberg Number Problem ($Wi \to 0.5$).
  - **Capitolo 2 - Sez 2.4:** Verificata la piena coerenza delle scale e dei residui PDE ($f_u, f_v, f_{\tau}$) con l'implementazione in `final_roll/src/physics.py`.
- **Prossimo Step Immediato:** Riorganizzazione finale e rifinitura della sezione 2.5 (Boundary Conditions e Four-Roll Mill: no-slip, gauge pressure e pinning).
