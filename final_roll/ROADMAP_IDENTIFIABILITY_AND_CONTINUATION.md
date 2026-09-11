# Roadmap Scientifica: Identificabilità Parametrica & Continuation in Weissenberg

**Progetto**: PINN Viscoelastica per Four-Roll Mill (Oldroyd-B)  
**Data di definizione**: 10 Settembre 2026  
**Stato**: In attesa dei nuovi dataset COMSOL  

---

## 1. Obiettivo e Cambio di Paradigma

Non cerchiamo più di forzare la convergenza della Fase 2 ad ogni costo sull'unico punto $\beta = 0.10$, ma utilizziamo le prossime simulazioni COMSOL per **mappare la regione dello spazio dei parametri $(\beta, Wi)$ in cui il problema inverso è fisicamente e numericamente identificabile**.

Questo trasforma una difficoltà locale di ottimizzazione in un contributo scientifico e metodologico centrale per la tesi:
1. Dimostrare **perché** regimi fortemente polymer-dominated ($\beta \le 0.10$) rendono degenerata l'identificazione di $\eta_s$ nel bilancio di quantità di moto.
2. Identificare la soglia di $\beta$ che rende il segnale del solvente $\eta_s \nabla^2 \mathbf{u}$ sufficientemente distinto dagli errori residui dello stress $\boldsymbol{\tau}_p$ e della pressione $p$.
3. Implementare un **metodo di continuazione / transfer learning su $Wi$** per scalare la soluzione di Fase 1 verso alte viscoelasticità in modo stabile e rapido.

---

## 2. Riferimento Letteratura: ViscoelasticNet (Thakur et al., 2024)

Dal paper di Thakur, Raissi, Ardekani (*JNNFM 2024*), i parametri impiegati per dimostrare l'identificazione nei modelli Oldroyd-B non sono mai estremi come $\beta = 0.10$:

| Setup | Geometria | $\lambda$ [s] | $\eta_p$ [Pa·s] | $\eta_s$ [Pa·s] | $\beta = \frac{\eta_s}{\eta_s+\eta_p}$ | $\frac{\eta_p}{\eta_{tot}}$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Oldroyd-B | Stenosi 2D | $0.050$ | $0.008$ | $0.010$ | **$0.556$** | $0.444$ |
| Oldroyd-B #1 | Cross-Slot | $0.005$ | $0.010$ | $0.010$ | **$0.500$** | $0.500$ |
| Oldroyd-B #2 | Cross-Slot | $0.015$ | $0.010$ | $0.020$ | **$0.667$** | $0.333$ |
| Oldroyd-B #3 | Cross-Slot | $0.010$ | $0.025$ | $0.020$ | **$0.444$** | $0.556$ |

**Conclusione di letteratura**: In tutti i benchmark Oldroyd-B convalidati, il solvente contribuisce tra il **44% e il 67%** della viscosità totale ($\beta \ge 0.44$).

---

## 3. Matrice delle Simulazioni COMSOL

Tutte le simulazioni manterranno rigorosamente costante la **viscosità totale**:
$$\eta_0 = \eta_s + \eta_p = 1.0 \text{ Pa}\cdot\text{s}$$
In questo modo il numero di Reynolds globale $Re = \frac{\rho U H}{\eta_0}$ e le scale idrodinamiche globali rimangono invariati; varia unicamente la ripartizione viscoelastica.

### Asse 1: Identificabilità Fase 2 (Viscosity Ratio Sweep a $\lambda = 0.05\text{ s}$)

| ID Dataset | File COMSOL | $\lambda$ [s] | $\eta_s$ [Pa·s] | $\eta_p$ [Pa·s] | $\beta$ | Priorità & Scopo |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **DS_A** | `4_roll_mill.csv` | $0.05$ | $0.10$ | $0.90$ | $0.10$ | **Baseline attuale** (F1 buona, F2 mal condizionata) |
| **DS_B** | `4_roll_mill_beta030.csv` | $0.05$ | $0.30$ | $0.70$ | **$0.30$** | **Priorità 1**: Prevalenza polimerica, ma segnale solvente 3x |
| **DS_C** | `4_roll_mill_beta050.csv` | $0.05$ | $0.50$ | $0.50$ | **$0.50$** | **Priorità 2**: Caso bilanciato 1:1 (omologo a Thakur) |
| **DS_D** | `4_roll_mill_beta070.csv` | $0.05$ | $0.70$ | $0.30$ | $0.70$ | *Opzionale*: Solo se B e C evidenziassero ancora criticità |

### Asse 2: Continuation in Weissenberg (fissato il miglior $\beta$, es. $\beta = 0.30$ o $0.50$)

| ID Dataset | $\lambda$ [s] | Stima $Wi = \lambda \dot{\gamma}_{char}$ | Ruolo nel Transfer Learning |
| :---: | :---: | :---: | :--- |
| **DS_LAM_1** | $\lambda_1$ (es. $0.01$) | $Wi \sim 0.2$ | **Warm-up**: Soluzione quasi-newtoniana, convergenza facile |
| **DS_LAM_2** | $\lambda_2$ (es. $0.025$) | $Wi \sim 0.5$ | **Step intermedio**: Transfer learning da checkpoint 1 |
| **DS_LAM_3** | $\lambda_3 = 0.05$ | $Wi \sim 1.0$ | **Target finale**: Transfer learning da checkpoint 2 |

*(I valori esatti di $\lambda$ saranno calibrati dopo aver misurato $\dot{\gamma}_{char}$ dal dataset attuale).*

---

## 4. Protocollo di Diagnostica Offline Preventiva (Zero Training a Vuoto)

Prima di avviare qualsiasi addestramento PINN sui nuovi dataset, eseguire lo screening offline calcolando sui nodi di COMSOL:
- Termine solvente: $\mathbf{A} = \eta_s \nabla^2 \mathbf{u}$
- Termine polimerico: $\mathbf{B} = \nabla \cdot \boldsymbol{\tau}_p$
- Gradiente di pressione: $\mathbf{C} = \nabla p$
- Convezione inerziale: $\mathbf{D} = \rho (\mathbf{u} \cdot \nabla \mathbf{u})$

### Metriche di Identificabilità:
1. **Rapporto di Segnale Globale ($L_2$ Ratio)**:
   $$R = \frac{\|\eta_s \nabla^2 \mathbf{u}\|_{L_2}}{\|\nabla \cdot \boldsymbol{\tau}_p\|_{L_2}}$$
   - Se $R < 0.10$: il parametro $\eta_s$ è coperto dall'errore residuo di $\boldsymbol{\tau}$ e dalla libertà di $p$.
   - Se $R \ge 0.30 - 0.50$: il parametro $\eta_s$ ha peso d'ordine primario nel bilancio, garantendo identificabilità.
2. **Correlazione Spaziale**:
   $$\text{corr}(\eta_s \nabla^2 \mathbf{u}, \nabla \cdot \boldsymbol{\tau}_p)$$
3. **Distribuzione Locale del Rapporto di Forza**:
   $$r(x, y) = \frac{|\eta_s \nabla^2 \mathbf{u}|}{|\nabla \cdot \boldsymbol{\tau}_p| + \epsilon}$$
4. **Sensibilità del Residuo di Quantità di Moto rispetto a $\eta_s$**:
   $$\frac{\partial \mathcal{R}_{mom}}{\partial \eta_s} = \nabla^2 \mathbf{u}$$

---

## 5. Protocollo di Continuation & Transfer Learning (Fase 1)

Quando si esplora la serie a $\lambda$ crescente:
1. Si addestra il modello per $\lambda = \lambda_{low}$ (Fase 1 con Adam + L-BFGS).
2. Si salva il checkpoint con i pesi addestrati $(\phi_{\psi}, \theta_{\tau})$.
3. Per il caso successivo $\lambda_{next} > \lambda_{low}$:
   - **Inizializzazione della Rete**:
     $$\theta_{NN}^{(k+1)} \longleftarrow \theta_{NN}^{(k)}$$
     Le reti `model_psi` e `model_tau` partono dai pesi già addestrati (cinematica a 4 vortici e topologia tensoriale già orientate).
   - **Reset Rigoroso dei Parametri Fisici**:
     $$\lambda^{(k+1)} \longleftarrow \lambda_{guess}^{(k+1)} = \lambda_{true}^{(k+1)} \cdot 0.80$$
     $$\eta_p^{(k+1)} \longleftarrow \eta_{p,guess}^{(k+1)} = \eta_{p,true}^{(k+1)} \cdot 0.80$$
     I parametri scalari **non ereditano** il valore convergente precedente, ma ripartono dal loro guess perturbato per dimostrare oggettivamente la capacità di ritrovare il nuovo valore vero.

---

## 6. Procedura Operativa all'Arrivo dei Dataset

Non appena i file COMSOL saranno esportati:
1. **Posizionamento**: Salvare i file in `COMSOL/4roll/` con nomenclatura coerente (`4_roll_mill_beta030.csv`, `4_roll_mill_beta050.csv`, ecc.).
2. **Esecuzione Diagnostica Offline**: Lanciare lo script diagnostico per estrarre $R, \dot{\gamma}_{char}, Wi$ e confrontare i dataset.
3. **Selezione del Dataset Promettente**: Scegliere il dataset con il miglior trade-off di segnale.
4. **Avvio Training Fase 1 Inversa**: Ottimizzare $\lambda$ e $\eta_p$.
5. **Avvio Training Fase 2 Inversa**: Con $\boldsymbol{\tau}$ congelato e $\psi$ mobile, ottimizzare $p$ e scoprire $\eta_s$.
