# Thesis Writing Tracker & Roadmap (10 Giorni - 2h/giorno)

**Obiettivo:** Finalizzazione della tesi magistrale in Ingegneria Chimica (*Physics-Informed Neural Networks per fluidi viscoelastici nel Four-Roll Mill*).  
**Formato sessioni:** Blocchi operativi mirati da max 2 ore al giorno (task modulari da 30-45 min).  
**Ultimo aggiornamento:** 2026-09-16  
**Stato Attuale:** **Giorno 1 (G1) - In corso**

---

## Roadmap e Stato Avanzamento

- [x] **G1: Ristrutturazione Cap 2 (Fluidodinamica)**
  - File: `Chapters/2.Fluid-dynamics.tex`
  - Output: Indice dettagliato con raccordo al Four-Roll Mill.
  - Stato: Completato.
- [x] **G2: Revisione testo Cap 2 (Parte 1)**
  - File: `Chapters/2.Fluid-dynamics.tex`
  - Output: Sezioni 2.1 e 2.2 chiuse (Governing equations, decomposizione dello sforzo).
  - Stato: Completato.
- [x] **G3: Chiusura Cap 2 (Parte 2)**
  - File: `Chapters/2.Fluid-dynamics.tex`
  - Output: Sezioni 2.3, 2.4 e 2.5 chiuse (Giesekus, PTT, Oldroyd-B, cinematica Four-Roll Mill con equazioni e citazioni, BCs).
  - Stato: Completato.
- [x] **G4: Ristrutturazione e revisione Cap 3 (PINN - Parte 1)**
  - File: `Chapters/3.PINNs.tex`
  - Output: Fondamenti Deep Learning, Universal Approximation, limiti data-driven, paradigma PINN.
  - Stato: Completato.
- [x] **G5: Chiusura Cap 3 (PINN - Parte 2)**
  - File: `Chapters/3.PINNs.tex`
  - Output: Deep dive training classico vs PINN (AD coordinate, assenza overfitting, meshless), Raissi et al., ViscoelasticNet, literature gap.
  - Stato: Completato.
- [x] **G6: Cap 4: Metodologia e Setup**
  - File: `Chapters/4.Results.tex`
  - Output: Recap matematico stream function, loss multiobiettivo, architettura 3-head SiLU, training a stadi (Phase 1 e Phase 2 con micro-unfreezing).
  - Stato: Completato.
- [x] **G7: Cap 4: Validazione COMSOL e Problema Diretto**
  - File: `Chapters/4.Results.tex`
  - Output: Studio convergenza di griglia con cutlines, accuratezza diretta $L_2 < 2\%$, calibrazione gauge pressione $\nabla p < 3.5\%$.
  - Stato: Completato.
- [ ] **G8: Cap 1: Introduzione e Motivazione**
  - File: `Chapters/1.Introduction.tex` (da strutturare/creare)
  - Output: Stesura completa: traiettoria da "limiti reometria classica" a "PINN nel four-roll mill".
- [x] **G9: Cap 4: Risultati Inversi, Mesh Independence, TL e Degenerazione PTT**
  - File: `Chapters/4.Results.tex`
  - Output: 4 benchmark inversi, grid independence (5k superiore a 125k), transfer learning (-50% epoche), limite invarianza $\mu_s$ a bassi $Re$, spiegazione analitica degenerazione PTT e conservazione modulo elastico $G = \mu_p/\lambda$.
  - Stato: Completato.
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

### Sessione G1-G7, G9 - 2026-10-03
- **Obiettivo:** Allineamento integrale con le slide del seminario e finalizzazione di Capitoli 2, 3 e 4.
- **Azioni completate:**
  - **Bibliografia (`references.bib`):** Aggiunte 9 citazioni autorevoli con DOI: Goodfellow et al. (2016), Baydin et al. (2018), Fuller & Leal (1981), D'Avino et al. (2017), Matos et al. (2026), Giesekus (1982), Phan-Thien & Tanner (1977), Kingma & Ba (2014), Cybenko (1989).
  - **Asset grafici (`Latex/images/`):** Importate e collegate tutte le 9 figure ad alta risoluzione (vettoriali e PNG).
  - **Capitolo 2 (`2.Fluid-dynamics.tex`):** Completato con Giesekus, PTT, Oldroyd-B, residui 2D cartesiani unificati, cinematica analitica 4-roll mill ($E, \omega, \lambda_p$), punto di ristagno iperbolico ($t_{\mathrm{res}} \to \infty$), geometric ratios di D'Avino e boundary conditions con pressure pinning.
  - **Capitolo 3 (`3.PINNs.tex`):** Completato con Deep Learning fundamentals (SiLU, MLP, Adam Cosine Annealing, L-BFGS), Universal Approximation, limiti data-driven, Deep Dive training standard vs PINN (differenziazione automatica alle coordinate, meshless, assenza di overfitting grazie al vincolo induttivo della PDE), Raissi et al. (2019) (pressione nascosta) e ViscoelasticNet (Thakur et al. 2024).
  - **Capitolo 4 (`4.Results.tex`):** Creato ex-novo e completato: recap matematico ViscoelasticNet a 3 teste con $\psi$, convergenza COMSOL con cutlines, validazione problema diretto (Tabella 1, $L_2 < 2\%$, calibrazione pressione $\nabla p < 3.5\%$), problema inverso su 4 benchmark (Tabella 2, errori sotto l'1-6\%), grid independence con spiegazione della superiorità di 5k (Tabella 3), transfer learning (-50% epoche), limite invarianza $\mu_s$ a bassi $Re$, spiegazione analitica della degenerazione PTT e conservazione rigida del modulo elastico $G = \mu_p/\lambda$ (Tabella 4), e posizionamento PINN come virtual rheometer.
  - **Compilazione TeX Live:** Eseguita con successo: **0 errori, 0 citazioni mancanti, 0 riferimenti rotti**. Generato [Latex/TESI.pdf](file:///c:/Users/eaglw/Documents/PINN%20tesi/Latex/TESI.pdf).

