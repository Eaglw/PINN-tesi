# Method: Active Learning & Bayesian Experimental Design (DoE)

## Overview
Questo metodo formalizza l'applicazione di **Processi Gaussiani (GP)** e **Active Learning** per l'identificazione autonoma della **frontiera di convergenza e operabilità** delle Physics-Informed Neural Networks (PINNs) applicate a flussi viscoelastici nel Four-Roll Mill.

A differenza della classica **Bayesian Optimization (BO)** mirata alla minimizzazione globale di una funzione costo (che tenderebbe a consigliare parametri fluidodinamici "facili" o quasi-newtoniani con basso $\text{Wi}$, dove l'errore è naturalmente minimo), questo protocollo implementa una **Level Set Estimation (LSE)** (o *Active Contour Finding*). L'obiettivo primario è campionare lo spazio dei parametri attorno alla transizione critica dove la PINN rischia la divergenza o l'identificazione errata dei parametri a causa dell'[[High_Weissenberg_Number_Problem]] (HWNP).

---

## 1. Definizione Matematica dello Spazio dei Parametri

Il dominio di esplorazione è definito dal vettore di parametri fisici e numerici:
$$\mathbf{x} = \big[ \lambda,\, \eta_p,\, \eta_s,\, \alpha,\, \varepsilon,\, \log_{10}(N_{\text{pts}}) \big] \in \mathbb{R}^6$$

| Parametro | Descrizione Fisica | Range Esplorato | Note Reologiche e Numeriche |
|---|---|---|---|
| **$\lambda$** | Tempo di rilassamento / Weissenberg proxy | $[0.05,\, 1.20]\,\mathrm{s}$ | Oltre $\lambda \approx 0.3 - 0.5$ insorgono gradienti estremi nello stagnation point. |
| **$\eta_p$** | Viscosità polimerica | $[0.10,\, 0.95]\,\mathrm{Pa\cdot s}$ | Con $\eta_s = 1.0 - \eta_p$ (viscosità totale normalizzata $\eta_{tot} = 1.0\,\mathrm{Pa\cdot s}$). |
| **$\eta_s$** | Viscosità del solvente | $[0.05,\, 0.90]\,\mathrm{Pa\cdot s}$ | Quando $\eta_s \to 0$ si perde la regolarizzazione newtoniana $\eta_s \nabla^2 \mathbf{u}$. |
| **$\alpha$** | Mobilità di Giesekus | $[0.00,\, 0.50]$ | $\alpha=0$ per Oldroyd-B. $\alpha > 0$ introduce shear-thinning quadratico $-\frac{\alpha \lambda}{\eta_p}\boldsymbol{\tau}^2$. |
| **$\varepsilon$** | Distruzione reticolare PTT | $[0.00,\, 0.50]$ | Introduce rilassamento esponenziale $\exp\left(\frac{\varepsilon \lambda}{\eta_p}\text{tr}(\boldsymbol{\tau})\right)$. |
| **`mesh`** | Nodi spaziali FEM ($N_{\text{pts}}$) | $[5k,\, 125k]$ | $5k$ (5,086 nodi), $12k$, $29k$, $52k$, $88k$, $125k$ (125,000 nodi). |

---

## 2. Metrica Target di Convergenza e Modello Surrogato (GP)

### A. Errore Parametrico Massimo
Dai risultati storici registrati in `inverse_runs.csv`, estraiamo la severità dell'errore di inversione sui soli parametri attivi per il modello fluido considerato:
$$E_{\text{param}}(\mathbf{x}) = \max\Big( |\text{err}_\lambda\%|,\, |\text{err}_{\eta_p}\%|,\, |\text{err}_\alpha\%|,\, |\text{err}_\varepsilon\%| \Big)$$

Poiché l'errore spazia da frazioni percentuali ($0.1\%$) a ordini di grandezza elevati ($>70\%$), il Gaussian Process modella la scala logaritmica:
$$z(\mathbf{x}) = \log_{10}\big(\max(E_{\text{param}},\, 0.05)\big)$$

Fissata la **soglia critica di buona convergenza a $\gamma = 10\%$**, la soglia nello spazio latente vale:
$$\gamma_{\log} = \log_{10}(10.0) = 1.0$$

### B. Gaussian Process Regressor con ARD
Il modello surrogato impiega un kernel **Matérn 5/2** con *Automatic Relevance Determination* (ARD), che assegna una scala di lunghezza $\ell_i$ indipendente a ciascuna coordinata parametrica:
$$d(\mathbf{x}, \mathbf{x}') = \sqrt{\sum_{i=1}^D \left(\frac{x_i - x'_i}{\ell_i}\right)^2}$$
$$k(\mathbf{x}, \mathbf{x}') = \sigma_f^2 \left( 1 + \sqrt{5}d + \frac{5}{3}d^2 \right) \exp(-\sqrt{5}d) + \sigma_n^2 \delta(\mathbf{x}, \mathbf{x}')$$

La predizione per un punto candidato $\mathbf{x}_*$ fornisce la media posteriore $\mu(\mathbf{x}_*)$ e l'**incertezza epistemica** $\sigma(\mathbf{x}_*)$ derivate da decomposizione di Cholesky:
$$\mu(\mathbf{x}_*) = \mathbf{k}_*^T (K + \sigma_n^2 I)^{-1} \mathbf{y}, \qquad \sigma^2(\mathbf{x}_*) = k(\mathbf{x}_*, \mathbf{x}_*) - \mathbf{k}_*^T (K + \sigma_n^2 I)^{-1} \mathbf{k}_*$$

---

## 3. Funzione di Acquisizione di Frontiera: *Straddle Heuristic*

Per individuare con la massima efficienza i punti a cavallo della convergenza, utilizziamo l'euristica di **Straddle** (Bryan et al., 2005):
$$a(\mathbf{x}) = \beta_{\text{exp}} \cdot \sigma(\mathbf{x}) - \big| \mu(\mathbf{x}) - \gamma_{\log} \big|$$
con parametro di esplorazione $\beta_{\text{exp}} = 1.96$ (intervallo di confidenza al 95%).

Questa formulazione bilancia due forze opposte:
1. **Esplorabilità pura ($\sigma(\mathbf{x})$)**: favorisce le aree dello spazio reologico inesplorate;
2. **Prossimità alla frontiera ($-|\mu(\mathbf{x}) - \gamma_{\log}|$))**: penalizza pesantemente sia i casi "facili" (dove la PINN converge con certezza e non apporta nuova conoscenza), sia i casi palesemente divergenti.

---

## 4. Algoritmo di Batching: *Kriging Believer*

Poiché le simulazioni CFD su COMSOL e i successivi addestramenti PINN richiedono tempi non trascurabili, suggerire un singolo punto alla volta è inefficiente. Tuttavia, prendere i primi $k$ punti ad acquisition massima porterebbe a un collasso spaziale nello stesso cluster.

L'algoritmo **Kriging Believer** implementato in `active_learning/gp_boundary.py` risolve il problema:
1. Identifica il primo candidato ottimo: $\mathbf{x}_1 = \arg\max_{\mathbf{x}} a(\mathbf{x})$.
2. Assegna una pseudo-osservazione pari alla predizione media attesa: $\hat{z}_1 = \mu(\mathbf{x}_1)$.
3. Aggiorna temporaneamente la matrice di covarianza inserendo $(\mathbf{x}_1, \hat{z}_1)$. L'incertezza $\sigma(\mathbf{x}_1)$ si azzera localmente.
4. Ricalcola l'acquisition function: il nuovo massimo $\mathbf{x}_2$ si troverà forzatamente in un'altra area critica non ridondante.
5. Itera fino a completare il batch ($k=3$).

---

## 5. Uso Operativo e Opzioni CLI

Il modulo risiede interamente in `active_learning/`:
```powershell
# Esecuzione standard non vincolata (Batch di 3 esperimenti)
.\venv\Scripts\python active_learning/suggest_batch.py

# Esecuzione con forzatura di diversità sui modelli costitutivi (Oldroyd-B, Giesekus, PTT)
.\venv\Scripts\python active_learning/suggest_batch.py --diverse-models

# Esecuzione focalizzata su un singolo modello
.\venv\Scripts\python active_learning/suggest_batch.py --model Oldroyd-B
```

Lo script produce sia l'output tabellare formattato secondo la convenzione di nomenclatura `4_roll_mill_L{lambda}-P{etap}-S{etas}-A{alpha}-E{eps}_M{mesh}.csv`, sia la mappa grafica 2D in `active_learning/plots/convergence_boundary_doe.png`.

---

## References & Back-links
- [[Viscoelastic_Training]]
- [[Viscoelastic_Fluids]]
- [[High_Weissenberg_Number_Problem]]
- [[ViscoelasticNet_Full model]]
- [[Mesh_Convergence_Protocol]]
