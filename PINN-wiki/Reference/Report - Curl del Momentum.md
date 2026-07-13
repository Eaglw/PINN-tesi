## 1. Il Problema Fisico e le Equazioni di Quantità di Moto (Momentum)

Consideriamo le equazioni della quantità di moto adimensionali per un fluido viscoelastico (modello costitutivo generico, es. Oldroyd-B, PTT o Giesekus). In forma vettoriale, l'equazione di bilancio della quantità di moto si scrive come:

$$
Re \left( \mathbf{u} \cdot \nabla \mathbf{u} \right) + \nabla p - \beta \nabla^2 \mathbf{u} - \nabla \cdot \boldsymbol{\tau} = 0 $$

Dove:
- $\mathbf{u} = (u, v)^T$ è il campo vettoriale di velocità (2D).
- $p$ è il campo scalare di pressione.
- $\boldsymbol{\tau} = \begin{pmatrix} \tau_{xx} & \tau_{xy} \\ \tau_{xy} & \tau_{yy} \end{pmatrix}$ è il tensore degli extra-stress polimerici.
- $Re$ è il numero di Reynolds.
- $\beta = \frac{\mu_s}{\mu_{\text{tot}}}$ è il rapporto di viscosità (frazione solvente).

Isolando il gradiente di pressione $\nabla p$, possiamo definire il **campo vettoriale di forze di momentum** $\mathbf{F} = (F_x, F_y)^T$:

$$
\nabla p = \mathbf{F}(\mathbf{u}, \boldsymbol{\tau})
$$

con:

$$
\mathbf{F}(\mathbf{u}, \boldsymbol{\tau}) = - Re \left( \mathbf{u} \cdot \nabla \mathbf{u} \right) + \beta \nabla^2 \mathbf{u} + \nabla \cdot \boldsymbol{\tau}
$$

---

## 2. La Condizione di Compatibilità del Rotore (Irrotazionalità di $\mathbf{F}$)

Dalle identità del calcolo vettoriale sappiamo che il rotore di un gradiente è identicamente nullo per qualsiasi campo scalare $p$ sufficientemente regolare ($p \in C^2$):

$$
\nabla \times (\nabla p) = \mathbf{0}
$$

Applicando l'operatore rotore ad entrambi i membri della relazione $\nabla p = \mathbf{F}$, otteniamo la **condizione di compatibilità per la pressione** (formulazione in vorticità):

$$
\nabla \times \mathbf{F}(\mathbf{u}, \boldsymbol{\tau}) = \mathbf{0}
$$

In un dominio bidimensionale (2D), il rotore di un campo vettoriale $\mathbf{F} = (F_x, F_y)^T$ si riduce a un campo scalare:

$$
\text{curl}(\mathbf{F}) = \frac{\partial F_y}{\partial x} - \frac{\partial F_x}{\partial y} = 0
$$

### Il concetto di Loss Floor
Se il campo di velocità appreso (tramite la stream function $\psi$) e il campo di stress $\boldsymbol{\tau}$ sono tali per cui $\text{curl}(\mathbf{F}) \neq 0$, allora il campo di forze $\mathbf{F}$ ammette, per il teorema di scomposizione di Helmholtz, una componente rotazionale (solenoide) non nulla $\mathbf{F}_{\text{rot}}$:

$$
\mathbf{F} = \nabla p_{\text{true}} + \mathbf{F}_{\text{rot}}, \quad \text{con} \quad \nabla \cdot \mathbf{F}_{\text{rot}} = 0 \quad \text{e} \quad \nabla \times \mathbf{F}_{\text{rot}} \neq 0
$$

Dato che il gradiente di pressione $\nabla p$ è puramente conservativo, la loss associata ai residui della momentum equation ($f_u, f_v$):

$$
\mathcal{L}_{\text{momentum}} = \frac{1}{2} \left\langle \|\nabla p - \mathbf{F}\|^2 \right\rangle = \frac{1}{2} \left\langle \|\nabla p - \nabla p_{\text{true}} - \mathbf{F}_{\text{rot}}\|^2 \right\rangle
$$

non potrà mai scendere sotto la soglia minima rappresentata dall'energia della componente rotazionale:

$$
\mathcal{L}_{\text{momentum}} \ge \frac{1}{2} \left\langle \|\mathbf{F}_{\text{rot}}\|^2 \right\rangle > 0
$$

Questo valore rappresenta il **loss floor** (limite inferiore di errore) che blocca l'ottimizzazione della pressione durante la Fase 2 del nostro staged training.

---

## 3. Implementazione Numerica nello Script `check_curl_F.py`

Per via della trasformazione geometrica e del riscalamento delle coordinate del dominio ($s = \frac{H_{\text{ref}}}{H_{\text{coord}}}$), le derivate spaziali calcolate via Autograd vengono moltiplicate per il fattore di scala $s$. 

Le componenti del vettore $\mathbf{F}$ calcolate nello script sono:

$$
F_x = -Re \cdot s \left( u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y} \right) + \beta \cdot s^2 \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} \right) + s \left( \frac{\partial \tau_{xx}}{\partial x} + \frac{\partial \tau_{xy}}{\partial y} \right)
$$

$$
F_y = -Re \cdot s \left( u \frac{\partial v}{\partial x} + v \frac{\partial v}{\partial y} \right) + \beta \cdot s^2 \left( \frac{\partial^2 v}{\partial x^2} + \frac{\partial^2 v}{\partial y^2} \right) + s \left( \frac{\partial \tau_{xy}}{\partial x} + \frac{\partial \tau_{yy}}{\partial y} \right)
$$

Il rotore di questo campo scalato viene stimato calcolando:

$$
\text{curl}(\mathbf{F}) = s \frac{\partial F_y}{\partial x} - s \frac{\partial F_x}{\partial y}
$$

---

## 4. Metriche di Valutazione per il Confronto

Lo script valuta e confronta le seguenti grandezze su un set di $N$ punti del dominio:

1. **Rotore Medio Assoluto**:
 $$
   \langle |\text{curl}(\mathbf{F})| \rangle = \frac{1}{N} \sum_{i=1}^N \left| \text{curl}(\mathbf{F})_i \right|
   $$

2. **Rotore Massimo Assoluto**:
 $$
   |\text{curl}(\mathbf{F})|_{\text{max}} = \max_{1 \le i \le N} \left| \text{curl}(\mathbf{F})_i \right|
   $$

3. **Intensità Media del Campo di Forze**:
 $$
   \langle |\mathbf{F}| \rangle = \frac{1}{N} \sum_{i=1}^N \sqrt{ (F_{x,i})^2 + (F_{y,i})^2 + \epsilon }
   $$

4. **Rapporto di Inconsistenza Fisica (Curl/Force Ratio)**:
 $$
   \text{Ratio} = \frac{\langle |\text{curl}(\mathbf{F})| \rangle}{\langle |\mathbf{F}| \rangle} \times 100\%
   $$

5. **Stima del Loss Floor Massimo (senza pressione)**:
 $$
   \mathcal{L}_{\text{floor}} = \frac{1}{2N} \sum_{i=1}^N \left( F_{x,i}^2 + F_{y,i}^2 \right)
   $$

---

## 5. Metodologia di Confronto: PINN Checkpoint vs. COMSOL Fit (Data-Driven)

Per capire se il rotore non nullo sia dovuto ad un limite numerico della differenziazione automatica (Autograd) o ad un'incoerenza fisica intrinseca appresa dalla PINN, lo script confronta due reti con la stessa architettura:

### Caso A: PINN Checkpoint (Fisica Parziale)
- Carica il modello addestrato in Fase 1 (`checkpoint_psi+tau_100k.pth`).
- In questa fase, la velocità (derivata di $\psi$) e lo stress polimerico ($\boldsymbol{\tau}$) sono stati addestrati con un accoppiamento debole o nullo della pressione.
- Calcola $\text{curl}(\mathbf{F})_{\text{PINN}}$.

### Caso B: Rete Data-Driven su COMSOL (Fitting ad Alta Precisione)
- Addestra una rete pulita esclusivamente sui dati esatti di COMSOL per la velocità $\mathbf{u}$ e lo stress $\boldsymbol{\tau}$ fino ad un errore L2 bassissimo:
$$
  E_{L2}(u) < 10^{-5}, \quad E_{L2}(\boldsymbol{\tau}) < 10^{-5}
  $$
- Questo modello rappresenta l'interpolatore ottimale dei dati esatti di COMSOL.
- Calcola $\text{curl}(\mathbf{F})_{\text{COMSOL\_fit}}$.

### Conclusioni Logiche del Confronto
- **Se** $\langle |\text{curl}(\mathbf{F})| \rangle_{\text{PINN}} \gg \langle |\text{curl}(\mathbf{F})| \rangle_{\text{COMSOL\_fit}}$, allora la deviazione irrotazionale della PINN è causata dalla mancanza di regolarizzazione fisica (mancanza del vincolo di pressione o del termine sul rotore) nella Fase 1.
- **Se** $\langle |\text{curl}(\mathbf{F})| \rangle_{\text{PINN}} \approx \langle |\text{curl}(\mathbf{F})| \rangle_{\text{COMSOL\_fit}}$, il rotore rilevato è un errore numerico intrinseco dovuto alla differenziazione successiva (fino al 4° ordine) tramite Autograd su dati discretizzati, definendo un limite di risoluzione insuperabile per questa architettura.
