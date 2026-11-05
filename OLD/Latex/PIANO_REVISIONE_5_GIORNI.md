# Piano Operativo di Revisione Tesi (Sprint 5 Giorni: 5–10 Ottobre 2026)

**Obiettivo:** Revisione critica, controllo qualità manuale (QC), completamento e consegna ufficiale della tesi di Laurea Magistrale in Ingegneria Chimica.  
**Titolo di lavoro:** *Physics-Informed Neural Networks per fluidi viscoelastici nel Four-Roll Mill*  
**Scadenza consegna:** Sabato 10 Ottobre 2026  
**Modalità di lavoro:** 1 capitolo al giorno (sessioni focalizzate da 1.5–2 ore) + 1 giornata di raccordo all'Università.  
**Stato al 5 Ottobre:** **GIORNO 1 IN CORSO — FOCUS: CAPITOLO 2**

---

## 📅 Tabella di Marcia (Day-by-Day)

| Data | Giorno | Capitolo Focus | Stato Bozza | Attività Utente | Supporto Antigravity |
| :--- | :---: | :--- | :---: | :--- | :--- |
| **05/10 (Lun)** | **G1** | **Cap. 2: Fluidodinamica e Reologia** |  Completo (33 KB) | Lettura critica e QC manuale su equazioni e fisica | Raccolta refusi/correzioni + Redazione bozza Cap. 1 |
| **06/10 (Mar)** | **G2** | **Cap. 3: Il Paradigma PINN** |  Completo (28 KB) | Lettura critica su AD, Raissi, ViscoelasticNet | Aggiornamenti LaTeX + Redazione bozza Cap. 5 |
| **07/10 (Mer)** | **G3** | **Cap. 4: Risultati e Discussione** *(Uni)* |  Completo (35 KB) | Controllo tabelle, mesh 5k vs 125k, confronto col relatore | Rifinitura grafici, cutlines e tabelle |
| **08/10 (Gio)** | **G4** | **Cap. 1: Introduzione e Motivazione** | ⏳ In bozza | Verifica filo conduttore: reometria classica $\to$ PINN | Integrazione revisioni e allineamento con Cap. 2-4 |
| **09/10 (Ven)** | **G5** | **Cap. 5: Conclusioni + Frontespizio** | ⏳ In bozza | Lettura conclusioni, verifica frontespizio, dedica e abstract | QC globale (0 warning, 0 citazioni rotte), compila PDF |
| **10/10 (Sab)** | **FINAL** | **Consegna Ufficiale** | 🎯 Chiusura | Check visivo finale del PDF e invio relatore | Congelamento archivio e repository |

---

## 🔍 Checklist Operativa per Capitolo

### Giorno 1 (Oggi, 05/10) — Capitolo 2: Fluidodinamica e Reologia
- [ ] **Continuità logica**: Il passaggio dalle equazioni generali di Navier-Stokes alla decomposizione dello sforzo elastico/viscoso è naturale?
- [ ] **Modelli costitutivi (Sez 2.2)**:
  - Definizione corretta della derivata conconvettiva superiore (upper-convected time derivative).
  - Formulazione Oldroyd-B, Phan-Thien-Tanner (PTT) con parametro $\epsilon$ ed esponenziale/lineare, Giesekus con parametro $\alpha$.
- [ ] **Cinematica Four-Roll Mill (Sez 2.3)**:
  - Descrizione del campo estensionale 2D e del punto di ristagno iperbolico centrale ($(0,0)$, con $t_{\mathrm{res}} \to \infty$).
  - Spiegazione del motivo per cui questo flusso è la sfida ideale per i fluidi viscoelastici (stress singularity/High Weissenberg Number Problem).
- [ ] **Numeri adimensionali e BC (Sez 2.4 - 2.5)**:
  - Definizione chiara di Weissenberg ($\text{Wi} = \lambda \dot{\gamma}$), Reynolds ($\text{Re}$) e rapporto di viscosità ($\beta = \mu_s/\mu_{\mathrm{tot}}$).
  - Correttezza delle condizioni al contorno: velocità sui rulli, pareti con no-slip, gauge di pressione (pressure pinning).

---

### Giorno 2 (06/10) — Capitolo 3: Il Paradigma PINN
- [ ] **Fondamenti Deep Learning**: MLP, funzioni di attivazione (SiLU), ottimizzatori combinati (Adam Cosine Annealing + L-BFGS a doppio stadio di precisione FP32/FP64).
- [ ] **Differenziazione Automatica (AD)**:
  - Differenza chiave tra AD rispetto alle coordinate spaziali $(x,y)$ e le differenze finite/mesh numeriche.
  - Perché la PINN non fa overfitting: il vincolo residuo della PDE agisce da regolarizzatore induttivo esatto.
- [ ] **Formulazione Stream Function**:
  - Dimostrazione che $u = \partial\psi/\partial y$ e $v = -\partial\psi/\partial x$ annullano identicamente la divergenza $\boldsymbol{\nabla} \cdot \boldsymbol{u} = 0$.
- [ ] **Letteratura e Frameworks**:
  - Il lavoro cardine di Raissi et al. (2019) e il concetto di pressione nascosta.
  - ViscoelasticNet (Thakur et al., 2024): inferenza di sforzo e parametri partendo da cinematica parziale.

---

### Giorno 3 (07/10) — Capitolo 4: Risultati e Discussione *(Giorno Uni)*
- [ ] **Validazione Problema Diretto**:
  - Tabella 1: errori relativi $L_2 < 2\%$ su velocità e sforzi.
  - Calibrazione gradiente di pressione $\nabla p < 3.5\%$ (nonostante il pressure gauge).
- [ ] **Problema Inverso (Benchmark e Parametri)**:
  - Tabella 2: stima accurata di $\lambda$ e viscosità per Oldroyd-B e Giesekus.
- [ ] **Mesh Independence (Superiorità 5k vs 125k)**:
  - Tabella 3: spiegazione del perché nel regime continuo la mesh a 5k punti regolarizza meglio senza subire il rumore ad alta frequenza della griglia fine.
- [ ] **Degenerazione Strutturale del PTT**:
  - Tabella 4: spiegazione analitica della conservazione rigida del modulo elastico $G = \mu_p / \lambda$.
- [ ] **Transfer Learning**:
  - Riduzione del 50% delle epoche di addestramento sfruttando pesi pre-addestrati.

---

### Giorno 4 (08/10) — Capitolo 1: Introduzione e Motivazione
- [ ] **Inquadramento**: Perché i fluidi viscoelastici sono cruciali nei processi industriali (polimeri, coating, microfluidica).
- [ ] **Limiti della reometria convenzionale**: Perdita di stabilità della superficie libera, rottura del menisco a taglio, impossibilità di isolare deformazione estensionale pura per tempi prolungati.
- [ ] **La soluzione**: Four-roll mill come reometro virtuale potenziato da PINN (misura indiretta non invasiva tramite velocimetria).
- [ ] **Obiettivi della tesi**: Elenco sintetico e strutturato dei traguardi scientifici.
- [ ] **Struttura del testo**: Guida sintetica ai capitoli successivi.

---

### Giorno 5 (09/10) — Capitolo 5: Conclusioni e Dettagli Editoriali
- [ ] **Conclusioni**: Sintesi obiettiva dei risultati sperimentali e numerici.
- [ ] **Limiti scientifici identificati**:
  - Non identifiabilità di $\mu_s$ a numeri di Reynolds prossimi allo zero (dominanza dello sforzo polimerico).
  - Degenerazione delle coppie di parametri per modelli complessi a taglio/estensione non saturati.
- [ ] **Prospettive future**: Estensione a flussi 3D, formulazione log-conformation per altissimi numeri di Weissenberg, regime transiente.
- [ ] **Frontespizio**: Inserimento relatore (Prof. M. M. Villone), correlatore (M. De Micco), matricola, candidato.
- [ ] **Abstract e Ringraziamenti**: Revisione testi italiani ed inglesi.

---

## 🎯 Le 5 Domande Guida per la Lettura Manuale

Durante la lettura, tieni a mente solo queste 5 domande:
1. **Chiarezza**: Chi legge senza aver visto il codice capisce il passaggio logico?
2. **Definizione Simboli**: I simboli matematici ($\boldsymbol{\tau}, \psi, \text{Wi}, \beta, \lambda$) sono definiti al primo utilizzo?
3. **Figure Richiamate**: Ogni immagine ha una frase che la introduce (es. *"Come mostrato in Figura 2.3..."*) prima di comparire?
4. **Numeri Coerenti**: Le tabelle del Capitolo 4 rispecchiano fedelmente i dati sperimentali?
5. **Tono Accademico**: Il linguaggio è rigoroso, formale e privo di modi di dire colloquiali?

---

## 📝 Log delle Modifiche e Revisioni Effettuate

*Questo spazio registra le annotazioni e le modifiche applicate durante le sessioni giornaliere.*

- **05/10 (Giorno 1 - Cap 2: Fluidodinamica e Reologia)**:
  -  Rimosso $\Omega \subset \mathbb{R}^3$, focalizzando l'apertura direttamente sulla meccanica dei fluidi.
  -  Sostituito uniformemente *Cauchy momentum equation* con *momentum equation*.
  -  Eliminati i tecnicismi prolissi su *solenoidal velocity field* e flussi isocori, mantenendo la chiara incompressibilità $\nabla \cdot \boldsymbol{u} = 0$.
  -  Corretto l'argomento sulla gravità (rimosso il ratio fuorviante; chiarito che $\boldsymbol{g}$ è ortogonale al piano 2D e assente in COMSOL).
  -  Sostituito *low-shear regime* con condizioni isoterme operative.
  -  Separato chiaramente lo stato stazionario dal regime laminare.
  -  Sostituito *two-fluid split* con *solvent-polymer stress decomposition* ed evitato *deviatoric stress* come sinonimo imperfetto di extra-stress.
  -  **UCTD e Gradiente**: definita esplicitamente la convenzione $(\nabla\boldsymbol{u})_{ij} = \partial u_i / \partial x_j$ e allineato l'operatore matriciale $\overset{\nabla}{\boldsymbol{\tau}}$ ai residui di `physics.py` (`upper_xx`, `upper_yy`, `upper_xy`), con nota di chiarimento sulla convenzione trasposta BSL.
  -  **Viscosità Estensionale Oldroyd-B**: corretta la formula inserendo il fattore $4\mu_p$ al numeratore (recupera rigorosamente il Trouton ratio $4\mu_0$ per $\dot{\epsilon} \to 0$).
  -  Ammorbidito il claim sulle PINN rispetto all'HWNP (evitano la griglia ma affrontano paesaggi di loss ripidi).
  -  Chiarita la funzione di rilassamento dello sforzo $Y(\text{tr}\,\boldsymbol{\tau})$ nel modello PTT e motivata la famiglia a 3 modelli per l'identificazione inversa.
  -  Distinto $Re_{\mathrm{scale}}$ (fondamentale per stabilizzare i gradienti durante l'inversione) da $Re_{\mathrm{phys}}$.
  -  **Eliminato completamente il numero di Deborah ($De$)**.
  -  Verificata la perfetta corrispondenza riga per riga tra i residui 2D cartesiani e l'implementazione in `physics.py`.
  -  Specificata la natura locale di $\lambda_p=0$ e chiarito il significato fisico degli autovalori $\pm E$ nel punto di sella.
  -  Sostituito $\boldsymbol{e}_z$ con rotazioni orarie/antiorarie e specificate le BC dello stress come dati informativi da simulazione di riferimento.
  -  **Pressure Pinning**: chiarito l'ancoraggio algebrico esatto (*hard algebraic anchor*) implementato in `CombinedModel` rispetto alla soft penalty.
  -  **Sezione 2.6 aggiunta**: tabella di raccordo con lo stato delle variabili (Osservate, Latenti, Parametri da identificare) per il passaggio al Capitolo 3.
  -  **Rifiniture di precisione (Pass 2)**:
     - Intro resa puramente fisica/matematica, rimandando i dettagli PINN al Cap. 3.
     - Introdotta la stream function $\psi$ ($u=\partial_y\psi, v=-\partial_x\psi$) a dimostrazione che $\nabla \cdot \boldsymbol{u} \equiv 0$ identicamente.
     - Pressione $p$ definita come *isotropic pressure field*.
     - $\omega$ definita come *local angular rotation rate* con vorticità scalare associata $-2\omega$, ed $E$ come *extension rate*.
     - Sostituito $\lambda_p$ con $\chi = (E-\omega)/(E+\omega)$, e corretta la rotazione dei rulli (adiacenti controrotanti, diagonali co-rotanti).
     - Oldroyd-B: termine *formulated/introduced* e Trouton ratio planare $Tr = \eta_E/\mu_0 \to 4$.
     - Giesekus e PTT descritti come estensioni non lineari distinte che condividono la base lineare.
     - Nondimensionalizzazione motivata dal miglioramento del condizionamento numerico.
     - Aggiunta distinzione tra $Wi$ globale e $\lambda E$ locale (soglia critica $2\lambda E \to 1$).
     - **Nuova Figura Regimi** ([four_roll_flow_regimes.pdf](images/four_roll_flow_regimes.pdf)) con 3 pannelli per $\chi = 1, 0, -1$.
     - **Nuova Figura Geometria Tesi** ([4roll_mill_geometry_thesis.pdf](images/4roll_mill_geometry_thesis.pdf)) vettoriale su sfondo bianco con quote $L, R, d_{\mathrm{roll}}$, assi $x,y$, origine $(0,0)$ e rotazioni $\Omega$.
     - Stress ai rulli marcati come *reference-solution boundary data*.
     - Pressure pin formalizzato come fissaggio dell'arbitraria costante additiva (*pressure gauge freedom*).
     - Aggiunta sezione di raccordo formale del problema forward vs inverse (dati $\boldsymbol{u}^{\mathrm{obs}}$, campi latenti, parametri $\boldsymbol{\theta}$).
  -  Compilazione LaTeX finale: **0 errori, 0 warning, riferimenti e figure risolti al 100%** ([TESI.pdf](TESI.pdf) aggiornato).
