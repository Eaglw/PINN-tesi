## 1. Modello Fisico e Equazioni Costitutive
Il modello di **Oldroyd-B** è una generalizzazione del modello di Maxwell che include una componente viscosa del solvente. Il tensore degli sforzi extra totale $\mathbf{T}$ è definito come:

$$\mathbf{T} = \mathbf{\tau}_s + \mathbf{\tau}_p$$

Dove:
*   **$\mathbf{\tau}_s$ (Solvente):** Comportamento Newtoniano, $\mathbf{\tau}_s = 2\mu_s \mathbf{D}$, dove $\mathbf{D} = \frac{1}{2}(\nabla \mathbf{u} + \nabla \mathbf{u}^T)$.
*   **$\mathbf{\tau}_p$ (Polimero):** Segue l'equazione dell'Upper-Convected Maxwell Model (UCM):
    $$\mathbf{\tau}_p + \lambda \overset{\triangledown}{\mathbf{\tau}_p} = 2\mu_p \mathbf{D}$$
    con $\lambda$ che rappresenta il tempo di rilassamento e $\overset{\triangledown}{\mathbf{\tau}_p}$ la derivata convettiva superiore (Upper-Convected Derivative).



---

## 2. Definizione del Problema: Flusso di Poiseuille
Si considera un canale 2D di lunghezza $L$ e altezza $H$, con il flusso guidato da un gradiente di pressione costante $\frac{dp}{dx}$. Le assunzioni di flusso completamente sviluppato e stazionario implicano:
*   $\mathbf{u} = (u(y), 0)$
*   $\frac{\partial u}{\partial x} = 0$

### Profili Analitici
Il codice implementa le soluzioni esatte derivate dalle equazioni di conservazione:

1.  **Velocità ($u$):**
    Il profilo rimane parabolico, identico al caso Newtoniano, ma governato dalla viscosità totale $\mu_{tot} = \mu_s + \mu_p$:
    $$u(y) = \frac{4 u_{max}}{H^2} y (H - y)$$
    Dove $u_{max} = \frac{|\Delta P| H^2}{8 L \mu_{tot}}$.

2.  **Sforzi Polimerici ($\tau$):**
    A differenza dei fluidi Newtoniani, la viscoelasticità introduce tensioni normali dovute allo shear rate $\dot{\gamma} = \frac{du}{dy}$:
    *   **Taglio:** $\tau_{xy} = \mu_p \dot{\gamma}$
    *   **Tensione Normale:** $\tau_{xx} = 2 \lambda \mu_p \dot{\gamma}^2$
    *   **Componente trasversale:** $\tau_{yy} = 0$

3.  **Funzione di Corrente ($\psi$):**
    Ottenuta per integrazione di $u(y)$ per soddisfare l'equazione di continuità:
    $$\psi(y) = \frac{4 u_{max}}{H^2} \left( \frac{H y^2}{2} - \frac{y^3}{3} \right)$$

---

## 3. Metodologia di Generazione
Il dataset viene costruito seguendo questi step logici:

### A. Campionamento Spaziale
Vengono supportate due modalità di generazione dei punti $(x, y)$:
*   **Grid Sampling:** Una griglia regolare cartesiana di $n_x \times n_y$ punti.
*   **Sobol Sampling:** Una sequenza quasi-casuale a bassa discrepanza, ideale per l'addestramento di Physics-Informed Neural Networks (PINNs) in quanto copre il dominio in modo più uniforme rispetto al campionamento puramente casuale.

### B. Consistenza Fisica
Il codice garantisce la coerenza tra i parametri termofisici ($\mu_s, \mu_p, \lambda$), la geometria ($H, L$) e le condizioni al contorno ($p_{in}, p_{out}$). Se `u_max` non è fornito, viene derivato analiticamente per assicurare che il campo di velocità rispetti il bilancio di quantità di moto.

### C. Modello di Rumore
Per simulare dati sperimentali o incertezze, è possibile aggiungere rumore Gaussiano:
*   **Percentage:** $\sigma = \text{noise\_value} \times \text{max\_val}$ (es. il 1% della velocità massima).
*   **Absolute:** $\sigma = \text{noise\_value}$.

---

## 4. Struttura del Dataset
Il file generato (`.pt` o `.csv`) contiene:
*   **Coordinate:** `x`, `y`
*   **Variabili di Stato:** `u`, `v`, `p`, `psi`
*   **Tensori degli Sforzi:** `tau_xx`, `tau_xy`, `tau_yy`
*   **Ground Truth:** Versioni `_exact` senza rumore per il calcolo dell'errore di validazione.
*   **Metadata:** Parametri fisici come il numero di Weissenberg ($Wi = \frac{\lambda u_{max}}{H}$) implicitamente definito dai parametri inseriti.