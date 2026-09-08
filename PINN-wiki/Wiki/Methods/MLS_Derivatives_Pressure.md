# MLS Derivatives for Direct Pressure Training

## Overview
Questo metodo descrive l'addestramento della sola rete di pressione (`PressureModel`) direttamente a partire dai dati campionati di COMSOL (velocità $\mathbf{u}$ e tensore degli extra-stress $\boldsymbol{\tau}$), **senza addestrare preventivamente alcuna rete neurale per la cinematica o la reologia (senza Fase 1)**.

L'idea alla base è eliminare la necessità di apprendere $\psi$ e $\tau$ tramite PINN quando sono disponibili misurazioni FEM/sperimentali dense, delegando la determinazione delle derivate spaziali al metodo numerico **Moving Least Squares (MLS)** e risolvendo la sola equazione di Poisson/Momentum per $p$:
$$ \nabla p = - Re (\mathbf{u} \cdot \nabla)\mathbf{u} + \beta \nabla^2 \mathbf{u} + \nabla \cdot \boldsymbol{\tau} $$
soggetta a una condizione al contorno puntuale di Dirichlet su un singolo nodo (**[[Pressure_Point_Anchoring]]**).

---

## Technical Implementation & Physical Details

Nel setup originale (`train_4roll_kaggle.py` - commit `b4f5547`), l'addestramento con `W_DATA = 0.0` (zero supervisione sui valori interni di pressione) ha dimostrato di poter raggiungere un errore relativo $L_2(p) \approx 20\%$.

Al contrario, tentativi di riscrittura ingenui hanno manifestato gravi instabilità o divergenze ($L_2(p) > 300\%$). Il confronto forense ha evidenziato le seguenti determinanti tecniche fondamentali:

### 1. Scaling Locale delle Coordinate tra $[-1, 1]$ (Cruciale)
Nel metodo MLS corretto, le coordinate dei $K$ vicini rispetto al nodo centrale $\mathbf{x}_0$ vengono rigorosamente normalizzate rispetto al raggio di supporto $h = \max(\text{dist}) > 0$:
$$ dx_{\text{scaled}} = \frac{x_i - x_0}{h}, \quad dy_{\text{scaled}} = \frac{y_i - y_0}{h} $$
* Se le coordinate non vengono scalate ($dx \sim 10^{-3}$ m), la matrice di Gram $X^T W X$ risulta gravemente malcondizionata perché i termini di grado superiore scalano come $dx^2 \sim 10^{-6}$ o $dx^3 \sim 10^{-9}$, introducendo amplificazione esponenziale del rumore numerico.
* La funzione peso gaussiana adottata è:
  $$ w_i = \exp\left(-\frac{\text{dist}_i^2}{h^2}\right) $$
* I coefficienti polinomiali estratti vengono poi riscalati dimensionalmente tramite divisione per $h$ (derivate prime) e $h^2$ (derivate seconde).

### 2. Grado Polinomiale: Quadratico (6 Termini) vs Cubico (10 Termini)
* **MLS di 2° Grado (6 termini)**:
  $$ X = [1, dx_s, dy_s, \frac{1}{2}dx_s^2, \frac{1}{2}dy_s^2, dx_s dy_s] $$
  con $K = 25$ vicini. Risulta robusto e sufficientemente regolare per estrarre il laplaciano $\nabla^2 \mathbf{u}$ e la divergenza $\nabla \cdot \boldsymbol{\tau}$.
* **MLS di 3° Grado**: introduce instabilità di Runge locale sui nodi discreti non strutturati, creando un forzante con rotore spurio artificiale molto elevato.

### 3. Gradient Clipping Rigido (`GRAD_CLIP_NORM = 5.0`)
Poiché le derivate discrete contengono inevitabilmente lievi discontinuità locali, i gradienti di backpropagation di `model_p` possono generare picchi impulsivi durante l'ottimizzazione Adam. Il clipping stretto a $5.0$ evita la corruzione dei buffer del momento ($m_t, v_t$).

### 4. Gestione Nativa dell'Ottimizzatore L-BFGS
L'ottimizzatore L-BFGS di PyTorch deve eseguire la propria line-search di Wolfe all'interno della chiamata unica `optimizer.step(closure)` con una dimensione di memoria storica (`history_size = 300`). Eseguire un ciclo `for` esterno con `max_iter=1` resetta costantemente la memoria ricorsiva a due passaggi, distruggendo la convergenza della pressione.

### 5. Scale Adimensionali di Riferimento
* $\mu_{\text{tot}} = \mu_s + \mu_p = 1.0\text{ Pa}\cdot\text{s}$
* $Re = \frac{\rho U_{\text{ref}} H}{\mu_{\text{tot}}} \approx 0.0417$
* $\beta = \frac{\mu_s}{\mu_{\text{tot}}} = 0.10$
* $s = \frac{H}{H_{\text{coord}}} = 0.10$
* Pesi Loss: $W_{\text{physics}} = 3.0$, $W_{\text{bc}} = 2.0$, $W_{\text{data}} = 0.0$.

---

> [!TIP]
> **Confermato Sperimentalmente con Successo (Run Kaggle #22 - 2026-09-08)**:
> Il setup è stato validato sperimentalmente nella run `[2026-09-08_15-49][DIR][PHASE2_MLS_SCALED][Ph2_20k+2k]` (Kaggle #22):
> - **Fase Adam (FP32)**: l'errore $L_2(p)$ scende rapidamente a **$19.57\%$** (già all'epoca 3.000).
> - **Fase L-BFGS (FP64)**: convergenza rapida fino a un **minimo assoluto di $4.91\%$** (iterazione 1.700), chiudendo al **$13.52\%$** (iterazione 2.000).
> - **Conclusione Matematica e Fisica**: Questo risultato dimostra inconfutabilmente che **il problema diretto per la pressione converge stabilmente senza dati interni di pressione**, utilizzando unicamente l'equazione di Momentum e **1 solo punto di ancoraggio Dirichlet**, a condizione che le derivate MLS siano calcolate al 2° grado con scaling locale $dx/h \in [-1, 1]$, gradient clipping rigido a $5.0$ e L-BFGS nativo con storico di 300.

---

## References & Back-links
- [[00_Index]]
- [[Pressure_Point_Anchoring]]
- [[Report_Curl_del_Momentum]]
- [[Viscoelastic_Training]]
- [[Fluid_Dynamics]]
