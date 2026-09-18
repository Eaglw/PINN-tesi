# Method: Viscoelastic Mesh Convergence and Worst-Case Boundary Protocol

## Overview
The **Viscoelastic Mesh Convergence Protocol** defines the rigorous numerical methodology used to prove the **spatial grid independence** of COMSOL Multiphysics reference solutions in the **Four-Roll Mill**, and governs how these validated spatial resolutions are transferred to Physics-Informed Neural Network (PINN) training datasets.

In numerical rheology, conducting a comprehensive mesh convergence study across every permutation of rheological parameters ($\lambda, \eta_p, \eta_s, \alpha, \varepsilon$) is computationally prohibitive and scientifically redundant. Instead, this protocol establishes the **Worst-Case Limiting Principle (Principio del Caso Limite)**: by rigorously demonstrating grid convergence on the mathematically most singular and demanding configuration, that mesh resolution is formally proven sufficient for all less restrictive flow regimes and non-linear constitutive models.

---

## 1. Theoretical Foundation: The "Worst-Case" Principle

As established in [[High_Weissenberg_Number_Problem]], the spatial gradient severity in viscoelastic flows is governed by:
1. **The Weissenberg Number ($Wi = \lambda \frac{U_{\text{ref}}}{H_{\text{ref}}}$)**: Higher relaxation times $\lambda$ produce exponentially steep stress wakes and thin boundary layers ($\delta \propto Wi^{-1}$).
2. **Constitutive Regularization**:
   - **Oldroyd-B** lacks any stress-saturation mechanism ($\eta_E \to \infty$ at $\dot{\varepsilon} \to 1/(2\lambda)$), making it the most singular and mesh-sensitive model in existence.
   - **Giesekus** ($\alpha > 0$, $\delta \propto Wi^{-1/2}$) and **PTT** ($\varepsilon > 0$, $\delta \propto Wi^{-1/3}$) feature non-linear shear-thinning and bounded extensional viscosities, yielding inherently broader and more regular boundary layers.

```
       RIGOROUS GRID INDEPENDENCE TRANSFER (THE "APPEND" STRATEGY)

   [ Oldroyd-B at λ_max (e.g., 0.1 s - 0.2 s) ]
   Tested on: M1 (125k) -> M2 (88k) -> M3 (52k) -> M4 (25k)
                      │
                      ▼
   [ Relative L2 Difference < 0.5% - 1.0% ]
   Convergence Threshold Identified (e.g., M3 / 52k nodes)
                      │
                      ├──────────────────────────────────────────────┐
                      ▼                                              ▼
   [ Lower Elasticity Regimes ]                    [ Regularized Constitutive Models ]
   • Oldroyd-B (λ = 0.05 s)                        • Giesekus (α = 0.1 - 0.3)
   • Thicker boundary layers                       • PTT (ε = 0.1 - 0.25)
   • Smooth, benign gradients                      • Bounded extensional stress
   ==> INHERITS MESH AUTOMATICALLY                 ==> INHERITS MESH AUTOMATICALLY
       (Zero redundant CFD sweeps)                     (Zero redundant CFD sweeps)
```

---

## 2. COMSOL Discretization Levels

Four spatial discretization levels are evaluated for the 2D Four-Roll Mill geometry ($L \times H = 0.10\text{ m} \times 0.10\text{ m}$, roll radii $R = 0.015\text{ m}$):

| Mesh Level | COMSOL Preset | Approximate Node Count | Purpose in Convergence Study |
|:---|:---|:---|:---|
| **M1** | *Extremely Fine* | $\approx 125.000$ | Asymptotic reference benchmark ("truth") |
| **M2** | *Extra Fine* | $\approx 88.000$ | Intermediate high-resolution verification |
| **M3** | *Finer* | $\approx 52.000$ | Target production resolution |
| **M4** | *Fine* | $\approx 25.000 - 30.000$ | Coarse lower bound (identifies numerical breakdown knee) |

> [!IMPORTANT]
> **Lower Bound Constraint**: Meshes coarser than $\approx 15.000 - 20.000$ nodes fail to adequately discretize the narrow pinch gap between adjacent counter-rotating cylinders ($d_{\text{gap}} \approx 0.01\text{ m}$), leading to artificial streamline diffusion (DEVSS/GLS numerical dissipation) and under-predicting the peak extensional stress at the central saddle point $(0, 0)$ by $> 15\%$.

---

## 3. Quantitative Verification Protocol (Cut-lines & Metrics)

To prove grid independence without relying solely on solver residuals, three specific diagnostics must be evaluated across $M_k$:

### Diagnostic 1: Critical Shear Velocity Profile in Roll Gap
Extract velocity $u(y)$ along the vertical cut-line bisecting the upper roll gap ($x = 0.025\text{ m}, y \in [0.01, 0.04]\text{ m}$):
$$
E_{L_2}(u; M_k) = \frac{\|u_{M_k}(y) - u_{M_1}(y)\|_{L_2}}{\|u_{M_1}(y)\|_{L_2}}
$$
- **Criterion**: $E_{L_2}(u; M_k) < 0.5\%$.

### Diagnostic 2: Peak Extensional Stress at Central Saddle Point
Extract $\tau_{xx}(x)$ along the central horizontal outflow streamline ($y = 0, x \in [-0.02, 0.02]\text{ m}$):
$$
E_{\text{peak}}(\tau_{xx}; M_k) = \frac{|\tau_{xx, M_k}(0, 0) - \tau_{xx, M_1}(0, 0)|}{|\tau_{xx, M_1}(0, 0)|}
$$
- **Criterion**: $E_{\text{peak}}(\tau_{xx}; M_k) < 1.5\%$.

### Diagnostic 3: Global Grid Convergence Index (GCI)
Based on Roache's verification standard:
$$
\text{GCI}_{23} = \frac{F_s |\varepsilon_{23}|}{r^p - 1}
$$
where $r = (N_1 / N_2)^{1/d}$ is the effective refinement ratio, $p$ is the formal order of accuracy, and $F_s = 1.25$ is the safety factor.

---

## 4. Transfer to PINN Training Datasets & The Convergence Duality

### Decoupling of FEM Mesh from PINN Batching
The PINN does not train on all $125.000$ nodes simultaneously:
- Kinematic supervision: $N_{\text{data}} \approx 2.000 - 5.000$ velocity points ($u, v$).
- Collocation points: $N_{\text{coll}} \approx 10.000 - 20.000$ internal domain points.

### The Essential Distinction: FEM Grid Convergence vs PINN Representation Capacity
> [!IMPORTANT]
> **La Dualità di Convergenza (FEM vs PINN)**:
> Una distinzione metodologica rigorosa deve essere mantenuta tra:
> 1. **Convergenza della Discretizzazione FEM di Riferimento (COMSOL)**: dimostrare che la soluzione numerica a $12\text{k}$ nodi si discosta per meno dello $0.23\%$ dalla mesh a $125\text{k}$ nodi garantisce che la "verità di terra" (ground truth) è formalmente asintotica e priva di rumore di discretizzazione.
> 2. **Convergenza dell'Architettura PINN**: verificare se $12\text{k}$ punti di collocazione siano sufficienti a garantire la convergenza della Physics-Informed Neural Network rimane una **questione sperimentale aperta**. Le PINN sono soggette a [[Spectral_Bias]], patologie dei gradienti (gradient flow stiffness tra termini avvettivi e diffusivi) e difficoltà nell'approssimazione dei ripidi strati limite di stress a ridosso dei cilindri tramite funzioni di attivazione standard.
> 
> **Valore dell'Analisi**: Avere dimostrato la convergenza numerica di COMSOL a 12k nodi è fondamentale perché **isola completamente le dinamiche della PINN**: qualsiasi discrepanza o residuo finale osservato durante l'addestramento neurale non potrà essere imputato a carenze o artefatti della soluzione FEM di riferimento, ma sarà attribuibile unicamente alla capacità rappresentativa, all'ottimizzazione e all'architettura della rete.

---

## 5. Empirical Benchmark: COMSOL 125k vs 12k (Four-Roll Mill)

Lo studio quantitativo di convergenza è stato eseguito confrontando:
- **Mesh Fine di Riferimento ($M_1$)**: `4_roll_mill_L0.1-P0.5-S0.5-A0-E0_M125k.csv` ($125.456$ nodi)
- **Mesh Coarse ($M_4$)**: `4_roll_mill_L0.1-P0.5-S0.5-A0-E0_M12k.csv` ($12.760$ nodi)
- **Configurazione Reologica**: Fluido di Oldroyd-B ($\lambda = 0.1\,\mathrm{s},\ \eta_p = 0.5\,\mathrm{Pa\cdot s},\ \eta_s = 0.5\,\mathrm{Pa\cdot s},\ \alpha = 0,\ \varepsilon = 0$).
- **Geometria**: Dominio $50 \times 50\,\mathrm{mm}$ ($x, y \in [-0.025, 0.025]\,\mathrm{m}$), quattro cilindri rotanti di raggio $R = 5\,\mathrm{mm}$ centrati in $(\pm 10\,\mathrm{mm}, \pm 10\,\mathrm{mm})$.

### 5.1 Tabella Quantitativa delle Norme di Errore

Le discrepanze puntuali sono calcolate interpolando la soluzione continua a 125k su tutti i nodi fluidi della mesh 12k:

| Grandezza | Rel $L_2$ Error [%] | Rel $L_1$ Error [%] | Max Abs Error ($L_\infty$) | 99° Percentile Errore | Scarto Picchi [%] |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **$u$ [m/s]** | **0.021%** | 0.015% | $1.692 \times 10^{-5}\ \mathrm{m/s}$ | $4.008 \times 10^{-6}\ \mathrm{m/s}$ | **0.01%** |
| **$v$ [m/s]** | **0.021%** | 0.015% | $1.150 \times 10^{-5}\ \mathrm{m/s}$ | $3.901 \times 10^{-6}\ \mathrm{m/s}$ | **0.01%** |
| **$\tau_{xx}$ [Pa]** | **0.222%** | 0.169% | $1.775 \times 10^{-2}\ \mathrm{Pa}$ | $8.579 \times 10^{-3}\ \mathrm{Pa}$ | **0.05%** |
| **$\tau_{xy}$ [Pa]** | **0.227%** | 0.162% | $1.142 \times 10^{-2}\ \mathrm{Pa}$ | $7.880 \times 10^{-3}\ \mathrm{Pa}$ | **0.03%** |
| **$\tau_{yy}$ [Pa]** | **0.221%** | 0.168% | $1.508 \times 10^{-2}\ \mathrm{Pa}$ | $8.360 \times 10^{-3}\ \mathrm{Pa}$ | **0.03%** |

### 5.2 Invarianti ed Integrali di Flusso
- **Energia Cinetica Totale $E_k = \frac{1}{2}\int_\Omega (u^2 + v^2)\,dA$**:
  - $E_k^{125k} = 1.52639 \times 10^{-8}\,\mathrm{J}$, $E_k^{12k} = 1.52631 \times 10^{-8}\,\mathrm{J}$ $\longrightarrow$ **Discrepanza: 0.0055%**.
- **Traccia Totale dello Stress Elastico $\int_\Omega (\tau_{xx} + \tau_{yy})\,dA$**:
  - Traccia $^{125k} = 3.74543 \times 10^{-4}\,\mathrm{N}$, Traccia $^{12k} = 3.69643 \times 10^{-4}\,\mathrm{N}$ $\longrightarrow$ **Discrepanza: 1.308%**.

---

## 6. Risultati Grafici sulle Cutline e Mappe Spaziali

I profili 1D sono estratti lungo 3 direttrici chiave:
1. **Orizzontale ($y = 0$)**: attraversa il punto di ristagno $(0,0)$ e il canale centrale.
2. **Verticale ($x = 0$)**: asse di simmetria verticale.
3. **Obliqua ($y = x$)**: attraversa Roll 3 $(-10, -10\,\mathrm{mm})$, il punto di ristagno $(0,0)$ e Roll 1 $(+10, +10\,\mathrm{mm})$. L'interno dei cilindri solidi ($r \le 5\,\mathrm{mm}$) è mascherato rigorosamente con `NaN` ed evidenziato in grigio tratteggiato.

### 6.1 Grafici Individuali per Grandezza di Fase 1

![[cutline_comparison_u.png]]
*Figura 1: Profilo di velocità $u$ lungo le 3 cutline con subplot dei residui $|\Delta| = |u_{12k} - u_{125k}|$.*

![[cutline_comparison_v.png]]
*Figura 2: Profilo di velocità $v$ lungo le 3 cutline con subplot dei residui.*

![[cutline_comparison_tau_xx.png]]
*Figura 3: Profilo dello sforzo normale viscoelastico $\tau_{xx}$. Notare la perfetta sovrapponibilità nel plateau centrale estensionale e nei gradienti a parete.*

![[cutline_comparison_tau_xy.png]]
*Figura 4: Profilo dello sforzo di taglio $\tau_{xy}$. Annullamento esatto per simmetria sugli assi centrali e picco fedele a parete del rullo.*

![[cutline_comparison_tau_yy.png]]
*Figura 5: Profilo dello sforzo normale viscoelastico $\tau_{yy}$. Cattura del picco estremo a $+2.6\,\mathrm{Pa}$ a parete senza smoothing numerico.*

### 6.2 Sintesi Panoramica Multi-Campo

![[cutlines_summary_all_fields.png]]
*Figura 6: Matrice comparativa 5x3 di tutte le grandezze di Fase 1 (righe) lungo le tre cutline (colonne).*

### 6.3 Localizzazione 2D dell'Errore e Distribuzione Cumulativa

![[mesh_convergence_2d_error_maps.png]]
*Figura 7: Mappe 2D della discrepanza assoluta $|f_{12k} - f_{125k}|$. L'errore è confinato entro frazioni di millimetro a parete, con errore quasi nullo nel 98% del dominio.*

![[mesh_convergence_error_cdf.png]]
*Figura 8: Istogramma dell'errore relativo normalizzato e Funzione Cumulativa di Ripartizione (CDF), attestante che il 99% del dominio ha errore $< 0.21\%$ del range dinamico.*

---

## References & Back-links
- [[High_Weissenberg_Number_Problem]]
- [[ViscoelasticNet_Full model]]
- [[Viscoelastic_Training]]
- [[Sampling_Strategies]]
- [[Spectral_Bias]]
- [[COMSOL_Boundary_Extraction]]
- [[Owens_Phillips_Computational_Rheology]]

