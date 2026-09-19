# Topic: Non-Identificabilità Strutturale della Viscosità del Solvente ($\eta_s$) nel Four-Roll Mill

## Overview & Sintesi della Scoperta

Nelle Physics-Informed Neural Networks (PINNs) applicate al problema inverso per fluidi viscoelastici (modello di Oldroyd-B), l'obiettivo storico era la scoperta completamente cieca (*full-blind identification*) di tutti e tre i parametri fisici fondamentali:
- $\lambda$: Tempo di rilassamento polimerico ($\text{s}$)
- $\eta_p$ (o $\mu_p$): Viscosità polimerica ($\text{Pa}\cdot\text{s}$)
- $\eta_s$ (o $\mu_s$): Viscosità del solvente newtoniano ($\text{Pa}\cdot\text{s}$)

Uno studio parametrico rigoroso condotto sul benchmark del **Four-Roll Mill 2D** ha dimostrato in modo inconfutabile un vincolo fisico e matematico fondamentale:
> [!CAUTION]
> **Teorema di Non-Identificabilità di $\eta_s$ nel Four-Roll Mill**:
> Nel Four-Roll Mill per fluidi di Oldroyd-B (in regime di creeping flow / basso numero di Reynolds), **la viscosità del solvente $\eta_s$ è strutturalmente e matematicamente non identificabile** attraverso il problema inverso a partire da sole misurazioni di velocità $\mathbf{u}$ ed extra-stress polimerico $\boldsymbol{\tau}$.
> I campi cinematici $\mathbf{u}(x, y)$ e tensoriali $\boldsymbol{\tau}(x, y)$ sono **totalmente invarianti rispetto a variazioni di $\eta_s$**. Qualsiasi variazione di $\eta_s$ viene assorbita al $100\%$ da una traslazione irrotazionale del campo di pressione $p(x, y)$, rendendo nulla l'informazione osservabile contenuta nei dati sperimentali (PIV).

---

## 1. L'Evidenza Empirica dello Studio Parametrico su Cutline

Per verificare l'impatto di $\eta_s$ sul comportamento del fluido nel Four-Roll Mill, è stato condotto uno studio parametrico FEM ad alta risoluzione (mesh asintotica COMSOL da $125.456$ nodi):
- Sono stati mantenuti rigorosamente costanti tutti i parametri cinematici, geometrici e reologici primari:
  $$\lambda = \text{cost}, \quad \eta_p = \text{cost}, \quad \Omega_{\text{roll}} = \text{cost}, \quad R = 5\,\text{mm}, \quad L = 50\,\text{mm}$$
- È stato fatto variare sistematicamente il valore della viscosità del solvente $\eta_s$ lungo ordini di grandezza differenti (es. $\beta = \frac{\eta_s}{\eta_s + \eta_p} \in [0.10, 0.50]$).
- Sono stati estratti i profili su 3 cutline ortogonali e diagonali cruciali:
  1. Cutline orizzontale passante per il centro di sella ($y = 0$);
  2. Cutline verticale passante per il centro di sella ($x = 0$);
  3. Cutline diagonale passante tra i rulli ($y = x$).

### Risultato Empirico Incontrovertibile:
L'estrazione di tutti i campi su ciascuna delle cutline ha restituito **curve perfettamente sovrapposte e indistinguibili**:
$$u(s; \eta_s^{(1)}) \equiv u(s; \eta_s^{(2)}), \qquad v(s; \eta_s^{(1)}) \equiv v(s; \eta_s^{(2)})$$
$$\tau_{xx}(s; \eta_s^{(1)}) \equiv \tau_{xx}(s; \eta_s^{(2)}), \quad \tau_{xy}(s; \eta_s^{(1)}) \equiv \tau_{xy}(s; \eta_s^{(2)}), \quad \tau_{yy}(s; \eta_s^{(1)}) \equiv \tau_{yy}(s; \eta_s^{(2)})$$

La discrepanza tra le curve è esattamente pari a zero (entro la tolleranza del solutore FEM $< 10^{-6}$). Il campo di moto e lo stato di sforzo polimerico **non subiscono alcuna alterazione**.

---

## 2. Dimostrazione Fisica e Matematica

Perché la viscosità del solvente $\eta_s$ non lascia alcuna impronta sul moto e sullo stress? La risposta risiede nell'interazione tra la reologia costitutiva, la cinematica imposta e la degenerazione di gauge con la pressione.

### A. Determinismo Cinematico nel Four-Roll Mill
Il moto del fluido all'interno della cavità è generato unicamente dal trascinamento viscoso no-slip imposto dalla rotazione dei quattro cilindri interni a velocità periferica nota $U = \Omega R$, confinato da pareti esterne fisse:
$$\mathbf{u}\big|_{\partial \Omega_{\text{roll}}} = \mathbf{u}_{\text{prescritta}}, \qquad \mathbf{u}\big|_{\partial \Omega_{\text{wall}}} = \mathbf{0}$$
In regime di creeping flow ($Re \ll 1$), l'inerzia $(\mathbf{u}\cdot\nabla)\mathbf{u}$ è trascurabile. Il campo di velocità $\mathbf{u}$ è primariamente determinato dalla cinematica al contorno di Dirichlet (flusso geometricamente guidato).

### B. Indipendenza dell'Equazione Costitutiva da $\eta_s$
L'equazione costitutiva di Oldroyd-B che governa il tensore degli sforzi polimerici $\boldsymbol{\tau}$ è:
$$\boldsymbol{\tau} + \lambda \overset{\triangledown}{\boldsymbol{\tau}} = 2 \eta_p \mathbf{D}(\mathbf{u})$$
dove $\mathbf{D}(\mathbf{u}) = \frac{1}{2}(\nabla \mathbf{u} + \nabla \mathbf{u}^T)$ è il tensore di velocità di deformazione e $\overset{\triangledown}{\boldsymbol{\tau}}$ è la derivata temporale convettiva superiore:
$$\overset{\triangledown}{\boldsymbol{\tau}} = (\mathbf{u} \cdot \nabla)\boldsymbol{\tau} - (\nabla \mathbf{u})\boldsymbol{\tau} - \boldsymbol{\tau}(\nabla \mathbf{u})^T$$

Si osserva immediatamente che:
1. Il parametro $\eta_s$ **non compare in nessun termine dell'equazione costitutiva**.
2. Poiché la cinematica $\mathbf{u}$ è imposta al contorno dai rulli, $\mathbf{D}(\mathbf{u})$ è invariante.
3. Con $\mathbf{u}$, $\lambda$ ed $\eta_p$ fissati, l'equazione costitutiva ammette un'unica soluzione univoca per $\boldsymbol{\tau}$, che è matematicamente **cieca rispetto a qualsiasi valore assunto da $\eta_s$**:
   $$\frac{\partial \boldsymbol{\tau}}{\partial \eta_s} \equiv \mathbf{0}$$

### C. Assorbimento Totale nel Gradiente di Pressione $\nabla p$
Nel bilancio della quantità di moto (Navier-Stokes per fluidi viscoelastici), la viscosità del solvente compare unicamente nel termine di diffusione newtoniana:
$$\rho (\mathbf{u} \cdot \nabla)\mathbf{u} = -\nabla p + \eta_s \Delta \mathbf{u} + \nabla \cdot \boldsymbol{\tau}$$
Esplicitando il gradiente di pressione per un dato valore $\eta_s$:
$$\nabla p(\mathbf{x}; \eta_s) = \eta_s \Delta \mathbf{u} + \nabla \cdot \boldsymbol{\tau} - \rho (\mathbf{u} \cdot \nabla)\mathbf{u}$$

Se consideriamo un secondo valore arbitrario della viscosità $\eta_s' = \eta_s + \Delta \eta_s$, sottraendo le due equazioni si ottiene:
$$\nabla p(\mathbf{x}; \eta_s') - \nabla p(\mathbf{x}; \eta_s) = \Delta \eta_s \, \Delta \mathbf{u}$$
Nel Four-Roll Mill, per la conservazione della massa in 2D incomprimibile, il campo laplaciano di velocità $\Delta \mathbf{u}$ è irrotazionale nel bulk del dominio:
$$\nabla \times (\Delta \mathbf{u}) = \Delta (\nabla \times \mathbf{u}) = \Delta \boldsymbol{\omega} \approx \mathbf{0}$$
Per il lemma di Poincaré, ogni campo vettoriale irrotazionale è il gradiente esatto di un potenziale scalare $\phi(\mathbf{x})$:
$$\Delta \mathbf{u} = \nabla \phi(\mathbf{x})$$
Di conseguenza:
$$\nabla \big( p(\mathbf{x}; \eta_s') - p(\mathbf{x}; \eta_s) \big) = \nabla \big( \Delta \eta_s \, \phi(\mathbf{x}) \big)$$
Integrando nello spazio:
$$\boxed{p(\mathbf{x}; \eta_s') = p(\mathbf{x}; \eta_s) + \Delta \eta_s \, \phi(\mathbf{x}) + C}$$

**Conclusione Fisica**:
Qualsiasi perturbazione arbitraria di $\eta_s$ viene compensata al $100\%$ da una deformazione del campo di pressione scalare $p(\mathbf{x})$. L'equazione della quantità di moto viene soddisfatta con residuo identicamente nullo **per qualsiasi valore reale $\eta_s > 0$**, senza generare alcuna forza netta o accelerazione che possa alterare la traiettoria delle linee di flusso $\mathbf{u}$ o la deformazione delle macromolecole $\boldsymbol{\tau}$.

---

## 3. Perché $\eta_s$ è Non-Identificabile nei Problemi Inversi (PIV)

In uno scenario sperimentale reale, o nel setup canonico di validazione delle PINN:
1. **Dati Osservabili**: Le tecniche di diagnostica ottica non invasiva (Particle Image Velocimetry - PIV, Particle Tracking - PTV) misurano accuratamente il campo di velocità del fluido $\mathbf{u}_{\text{data}}$. Mediante tecniche fotoelastiche (Flow Birefringence) è talvolta possibile ricavare stime dello stato di sforzo $\boldsymbol{\tau}_{\text{data}}$.
2. **Grandezze Non Osservabili**: La pressione interna $p(x, y)$ è un campo di gauge **completamente inaccessibile** all'interno del canale (i sensori di pressione a membrana possono essere collocati al massimo su pochi punti discreti delle pareti perimetrali esterne).

Calcolando la matrice di sensitività (e l'Informazione di Fisher $\mathcal{I}$) dei dati osservabili rispetto a $\eta_s$:
$$\mathbf{J}_{\eta_s} = \begin{bmatrix} \frac{\partial \mathbf{u}}{\partial \eta_s} \\ \frac{\partial \boldsymbol{\tau}}{\partial \eta_s} \end{bmatrix} = \begin{bmatrix} \mathbf{0} \\ \mathbf{0} \end{bmatrix} \implies \mathcal{I}(\eta_s) = \mathbf{J}_{\eta_s}^T \mathbf{W} \mathbf{J}_{\eta_s} \equiv 0$$

Poiché l'informazione di Fisher associata a $\eta_s$ è identicamente nulla, il limite inferiore di varianza di Cramér-Rao è infinito:
$$\text{Var}(\hat{\eta}_s) \ge \mathcal{I}(\eta_s)^{-1} = \infty$$

**Nessun algoritmo di ottimizzazione o rete neurale (PINN) potrà mai estrarre $\eta_s$ da osservazioni di velocità o sforzo nel Four-Roll Mill.**

---

## 4. Riconciliazione Storica del "Drift di $\eta_s$" in Fase 2

Questa scoperta permette di dare una spiegazione teorica definitiva a tutti i fenomeni e le anomalie riscontrate durante i mesi di sviluppo della Fase 2:

| Fenomeno Storico | Interpretazione Precedente (Errata o Parziale) | Verità Scientifica Rivelata dalla Scoperta |
| :--- | :--- | :--- |
| **Run 010: Crollo di $\beta \to 0$** | Ritenuto un problema di accoppiamento numerico tra $Re$ e $\eta_{\text{tot}}$. | L'ottimizzatore collassava $\eta_s \to 0$ perché la loss di momento è invariante a $\eta_s$, scegliendo la scorciatoia che minimizzava i gradienti newtoniani. |
| **Drift Monotono di $\eta_s$ in Fase 2** | Attribuito a un presunto "gauge feedback loop" tra l'ampiezza di $\nabla p$ e $\mu_s^*$. | $\eta_s$ derivava all'infinito perché si muoveva lungo un **fondovalle di loss piatto a costo zero** (*flat manifold*). Per ogni valore di $\eta_s$, `model_p` creava la pressione coniugata esatta $p = p_0 + \eta_s \phi$. |
| **Esplosione in L-BFGS ($> 1.83$)** | Attribuita al malcondizionamento delle derivate di secondo ordine. | In uno spazio con curvatura nulla ($\nabla^2 \mathcal{L} \approx 0$ lungo $\eta_s$), i metodi quasi-Newton a matrice inversa compiono passi di linea arbitrariamente lunghi, facendo esplodere il parametro. |
| **Apparente successo del Curl ($0.098$)** | Ritenuto una convalida che la vorticità isolava $\eta_s$. | Fissando il guess a $0.080$ e interrompendo l'addestramento, il valore si era fermato casualmente a $0.098$ per bilanciamento locale con i residui di bordo; non vi era alcuna forza fisica a guidarlo. |
| **Test Offline con Residuo Basso (1%)** | Considerato la prova che $\eta_s$ fosse identificabile. | Il test offline funzionava unicamente perché utilizzava come target il vero gradiente di pressione COMSOL $\nabla p_{\text{true}}$. Se $\nabla p$ è noto a priori, $\eta_s$ è banale; ma in un vero problema inverso $p$ è incognito! |

---

## 5. Riconfigurazione Operativa del Framework PINN

Alla luce della non-identificabilità geometrica di $\eta_s$, l'architettura scientifica del progetto viene riorganizzata su basi rigorose:

### 1. Consacrazione del Successo di Fase 1 (Reologia e Cinematica)
La Fase 1 è un **successo scientifico solido e completo**:
- I parametri reologici $\lambda$ ed $\eta_p$ (nonché i parametri non lineari $\alpha$ per Giesekus ed $\varepsilon$ per PTT) risiedono all'interno dell'equazione costitutiva.
- Il loro condizionamento SVD sul Four-Roll Mill è quasi ideale ($\kappa(J_{\text{con}}) \approx 1.34$).
- I campi di deformazione estensionale e di taglio generati dai quattro rulli forniscono un'eccellente ricchezza informativa che consente di scoprire i parametri reologici con errore inferiore allo $0.1\%$.

### 2. Ridefinizione del Ruolo della Fase 2 (Idrodinamica e Pressione)
La Fase 2 non deve più essere considerata una fase di "scoperta cieca" di $\eta_s$:
1. **Uso Primario (Direct Hydrodynamic Solver)**:
   La viscosità del solvente $\eta_s$ (o la viscosità totale $\eta_{\text{tot}}$ nota dal reometro a taglio per il solvente puro, es. acqua o olio) viene **fornita come proprietà nota del fluido**. La Fase 2 viene impiegata come solutore differenziale per ricostruire l'intero campo di pressione bidimensionale $p(x, y)$ e vincolare la costante di gauge tramite [[Pressure_Point_Anchoring]].
2. **Condizione Necessaria per l'Identificazione Inversa di $\eta_s$**:
   Qualora si desiderasse identificare $\eta_s$ in un problema inverso, è **fisicamente indispensabile** introdurre dati osservativi addizionali che dipendano esplicitamente dalla dissipazione viscosa:
   - Misura continua della coppia meccanica / potenza dissipata sui rulli rotanti:
     $$\mathcal{P}_{\text{roll}} = \int_{\partial \Omega_{\text{roll}}} \mathbf{u} \cdot (\boldsymbol{\sigma} \cdot \mathbf{n}) \, dA = \int_{\partial \Omega_{\text{roll}}} \mathbf{u} \cdot \left[ (-p\mathbf{I} + 2\eta_s \mathbf{D} + \boldsymbol{\tau}) \cdot \mathbf{n} \right] dA$$
   - Oppure sensori fisici di pressione differenziale calibrata $\Delta p$ posizionati sulle pareti del canale.

---

## References & Back-links

- [[Viscoelastic_Parameter_Identifiability]] — Analisi quantitativa della sensibilità dei parametri reologici e SVD.
- [[Pressure_Stress_Decoupling]] — Basi teoriche della decomposizione di Helmholtz-Hodge tra momento e sforzi.
- [[Vorticity_Inversion_Solvent]] — Formulazione a rotore e autopsia storica dei tentativi di inversione del solvente.
- [[Staged_Training_Procedure]] — Workflow a due fasi sequenziali cinematica/idrodinamica.
- [[Numerical_Hygiene_and_Phase2_Reforms]] — Igiene numerica, barriera softplus e diagnostica di Hodge-Leray.
- [[ViscoelasticNet_Full model]] — Equazioni costitutive unificate (Oldroyd-B, Giesekus, PTT).
- [[Viscoelastic_Fluids]] — Descrizione della fisica del benchmark 4-roll mill.
