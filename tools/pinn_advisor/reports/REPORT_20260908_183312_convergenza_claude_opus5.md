# Analisi di convergenza per PINN viscoelastico in Four‑Roll Mill

Premetto una **diagnosi strutturale** che condiziona tutte le cinque risposte, perché individua un problema di *identificabilità* che nessuna scelta di iperparametri può risolvere.

---

## 0. Diagnosi preliminare: non‑identificabilità strutturale di λ in Fase 1

Il modello costitutivo Oldroyd‑B, in forma stazionaria, è

$$
\boldsymbol{\tau} + \lambda\left[\mathbf{u}\cdot\nabla\boldsymbol{\tau} - \mathbf{L}\boldsymbol{\tau} - \boldsymbol{\tau}\mathbf{L}^{\mathsf T}\right] = 2\mu_p\mathbf{D},
\qquad \mathbf{L}=\nabla\mathbf{u},\;\; \mathbf{D}=\tfrac12(\mathbf{L}+\mathbf{L}^{\mathsf T}).
$$

Il residuo costitutivo $\mathcal{R}_c[\boldsymbol\tau;\lambda,\mu_p]$ è **lineare in $\boldsymbol\tau$**, **lineare in $\mu_p$** e **lineare in $\lambda$** (perché $\varepsilon_{PTT}=\alpha_{Gsk}=0$). Ne discende un teorema banale ma devastante:

> **Se il campo $\boldsymbol\tau$ è rappresentato da una rete universale non vincolata e $\mathbf{u}$ è fissato dai dati, allora per *ogni* $\lambda>0$ esiste $\boldsymbol\tau^\star(\lambda)$ tale che $\mathcal{R}_c\equiv 0$.** Il minimo della sola loss costitutiva è quindi degenere lungo un'intera varietà $(\lambda,\mu_p,\boldsymbol\tau^\star(\lambda,\mu_p))$, e include il punto banale $\mu_p=0,\ \boldsymbol\tau\equiv 0$.

Conseguenze operative immediate per il vostro codice:

1. In Fase 1 con `w_mom = 0` e senza dati interni su $\boldsymbol\tau$, **l'unica sorgente di informazione su $\lambda$ e $\mu_p$ è `USE_ROLL_STRESS_BC`**. Disattivarlo rende il problema inverso mal posto (non solo mal condizionato). Questo risponde in parte alla Domanda 2: il peso `W_ROLL_STRESS` non è un iperparametro di regolarizzazione, è **il likelihood dell'intero problema inverso di Fase 1**.
2. La direzione $\mu_p \to 0$ è un attrattore spurio: `initialize_last_layer_zero(model_tau)` vi colloca esattamente sopra ($\boldsymbol\tau=0$) all'epoca 0. Con $\boldsymbol\tau=0$, $\partial\mathcal{L}_c/\partial\lambda = 0$ **esattamente** (il termine in $\lambda$ moltiplica $\boldsymbol\tau$ e le sue derivate). Questo è il plateau iniziale della Domanda 4: non è un problema di parametrizzazione log, è un **punto critico degenere**.
3. Poiché il residuo è *bilineare* in $(\lambda,\mu_p)$ dato $(\boldsymbol\tau,\mathbf{u})$, i due parametri **non vanno ottimizzati con Adam**: esiste la soluzione in forma chiusa (§4.3, *variable projection*).

---

## 1. Sorgenti matematiche di stiffness e disaccoppiamento di $\tau_{xx}$

### 1.1 Struttura spettrale dell'operatore costitutivo

Scrivendo il tensore simmetrico come vettore $\mathbf{t}=(\tau_{xx},\tau_{xy},\tau_{yy})^{\mathsf T}$, la parte algebrica dell'equazione è $\mathbf{M}(\lambda,\nabla\mathbf u)\,\mathbf t$ con

$$
\mathbf{M} = \mathbf{I} - \lambda
\begin{pmatrix}
2u_x & 2u_y & 0\\
v_x & u_x+v_y & u_y\\
0 & 2v_x & 2v_y
\end{pmatrix}
\;\overset{v_y=-u_x}{=}\;
\mathbf{I}-\lambda
\begin{pmatrix}
2u_x & 2u_y & 0\\
v_x & 0 & u_y\\
0 & 2v_x & -2u_x
\end{pmatrix}.
$$

Gli autovalori dell'operatore $\boldsymbol\tau\mapsto \mathbf L\boldsymbol\tau+\boldsymbol\tau\mathbf L^{\mathsf T}$ sono $\{2s,\,0,\,-2s\}$ con $s=\pm\sqrt{u_x^2+u_yv_x}=\pm\sqrt{-\det\mathbf L}$ autovalori di $\mathbf L$. Quindi

$$
\boxed{\ \sigma(\mathbf M)=\{\,1-2\lambda s,\ 1,\ 1+2\lambda s\,\},\qquad
\kappa(\mathbf M)=\frac{1+2\lambda s}{1-2\lambda s}\ \ \text{se } Q\equiv u_x^2+u_yv_x>0. }
$$

$Q$ è (a meno di una costante) il **parametro di Okubo–Weiss**: $Q>0$ regione dominata dalla deformazione (iperbolica), $Q<0$ dominata dalla vorticità. Questo è il risultato centrale:

* Nei **vortici** attorno ai quattro rulli $Q<0$, $s$ è immaginario puro, $|1\mp 2\lambda s|=\sqrt{1+4\lambda^2|s|^2}\ge 1$: l'operatore è **ben condizionato** (rotazione = "media" delle direzioni principali).
* Nell'intorno del **punto di stagnazione iperbolico** e lungo le separatrici $Q>0$ e $\kappa(\mathbf M)\to\infty$ per $\mathrm{De}_{loc}\equiv 2\lambda\sqrt Q\to 1$.

Nel flusso puramente estensionale $\mathbf u=\dot\varepsilon(x,-y)$ le tre componenti sono **esattamente disaccoppiate** con tassi di rilassamento lungo la traiettoria

$$
\underbrace{\tfrac1\lambda - 2\dot\varepsilon}_{xx},\qquad \underbrace{\tfrac1\lambda}_{xy},\qquad \underbrace{\tfrac1\lambda+2\dot\varepsilon}_{yy},
$$

e soluzioni stazionarie
$$
\tau_{xx}=\frac{2\mu_p\dot\varepsilon}{1-2\lambda\dot\varepsilon},\qquad
\tau_{yy}=\frac{-2\mu_p\dot\varepsilon}{1+2\lambda\dot\varepsilon},\qquad \tau_{xy}=0 .
$$

**La singolarità di Weissenberg critico è $\mathrm{Wi}_c=\lambda\dot\varepsilon = 1/2$** (catastrofe coil–stretch dei dumbbell hookeani infinitamente estensibili). Con $\lambda=0.05\,$s: $\dot\varepsilon_c=10\ \mathrm{s^{-1}}$. **Diagnostica obbligatoria da inserire subito**: mappa di $\mathrm{De}_{loc}(x,y)=2\lambda\sqrt{\max(Q,0)}$ e suo massimo, loggata a TensorBoard ad ogni checkpoint. Se durante l'ottimizzazione $\lambda$ transita oltre $\lambda_{\max}=0.5/\sqrt{Q_{\max}}$ il campo $\boldsymbol\tau^\star$ diverge, la loss esplode e Adam produce un NaN o un salto irrecuperabile.

### 1.2 Il modo omogeneo singolare al punto di stagnazione

Lungo l'asse di uscita $y=0$, con $u=\dot\varepsilon x$:

$$
\lambda\dot\varepsilon\,x\,\tau_{xx}' + (1-2\lambda\dot\varepsilon)\tau_{xx}=2\mu_p\dot\varepsilon
\;\Longrightarrow\;
\tau_{xx}=\underbrace{\frac{2\mu_p\dot\varepsilon}{1-2\lambda\dot\varepsilon}}_{\text{particolare}} + \;C\,x^{-\left(\frac{1}{\lambda\dot\varepsilon}-2\right)} .
$$

Per $\lambda\dot\varepsilon<1/2$ l'esponente è negativo: **esiste un modo omogeneo illimitato in $x=0$**. Il problema ben posto richiede $C=0$, cosa che in CFD si ottiene automaticamente dall'*upwinding* lungo le caratteristiche. Una PINN, che minimizza il residuo in norma $L^2$ senza rispettare le caratteristiche, **eccita parzialmente questo modo**: è la causa fisica del tipico "spike" o della struttura oscillante che si osserva sull'asse di uscita. Rimedi: (i) pesatura *causale* lungo linee di corrente (§2.3); (ii) collocazione infittita in $|x|<\delta$; (iii) penalizzazione esplicita di $\partial_x\tau_{xx}$ nell'intorno del punto di stagnazione (dove la soluzione fisica è localmente costante).

### 1.3 Le tre strategie di disaccoppiamento, in ordine di efficacia

**(A) Equilibrazione di riga del residuo (row scaling) — costo nullo, effetto immediato.**
Il vostro codice divide tutti e tre i residui per lo *stesso* `tau_scale`. Poiché $\tau_{xx}\sim(1-\mathrm{De})^{-1}$ e $\tau_{yy}\sim(1+\mathrm{De})^{-1}$, i tre canali hanno ampiezze che differiscono di $\kappa(\mathbf M)$. Il teorema di van der Sluis garantisce che lo scaling di riga che equalizza le norme è ottimale entro $\sqrt{n}$ rispetto al condizionamento minimo. Implementazione:

```python
# dentro compute_residuals, ramo costitutivo
with torch.no_grad():
    # precondizionatore diagonale locale (detached: non altera il minimizzatore)
    s2 = u_x**2 + u_y*v_x                      # = -det(L)
    s  = torch.sqrt(torch.clamp(s2, min=0.0))  # tasso di strain iperbolico
    De = 2.0 * Wi_local * s                    # Wi_local = lam * (H_ref/U_ref)^-1 ... coerente ai vostri adim.
    d_xx = torch.clamp(1.0 - De, min=0.05)     # floor per evitare 1/0
    d_xy = torch.ones_like(d_xx)
    d_yy = 1.0 + De
    # scala di riferimento fisica per ogni canale
    s_xx = self.tau_scale_xx; s_xy = self.tau_scale_xy; s_yy = self.tau_scale_yy

f_txx = f_txx / (s_xx * d_xx)
f_txy = f_txy / (s_xy * d_xy)
f_tyy = f_tyy / (s_yy * d_yy)
```

Questo **è** il disaccoppiamento richiesto: nel frame principale $\mathbf M$ è diagonale e la divisione per $|1\mp2\lambda s|$ rende i tre canali equi‑sensibili. Nota importante: il precondizionatore **deve essere `detach()`ato**, altrimenti introducete una dipendenza spuria da $\lambda$ nel funzionale e biassate la stima.

**(B) Formulazione log‑conformation (Fattal–Kupferman) — la modifica strutturalmente corretta.**
Ponendo $\boldsymbol\tau=\frac{\mu_p}{\lambda}(\mathbf c-\mathbf I)$ con $\mathbf c$ SPD e $\boldsymbol\Psi=\log\mathbf c$, la soluzione estensionale diventa

$$
\Psi_{xx} = -\ln(1-2\lambda\dot\varepsilon),\qquad \Psi_{yy}=-\ln(1+2\lambda\dot\varepsilon).
$$

**Il polo si trasforma in un logaritmo**: per $\mathrm{De}=0.9$, $\tau_{xx}$ è amplificato $10\times$ mentre $\Psi_{xx}=2.30$. La dinamica in $\ln$ è additiva anziché moltiplicativa: la rete deve rappresentare un campo con *dynamic range* di $O(1)$ invece di $O(\kappa)$, il che mitiga direttamente il *spectral bias* sul filamento birifrangente. In più si garantisce $\mathbf c\succ 0$ (cioè $\tau_{xx}>-\mu_p/\lambda$), vincolo che una rete a 3 uscite libere viola sistematicamente durante il transitorio, generando instabilità di Hadamard.

Per 2×2 simmetrico l'esponenziale è in forma chiusa e differenziabile:

```python
def sym_expm_2x2(a, b, c):                  # Psi = [[a,b],[b,c]]
    m = 0.5*(a+c); d2 = (0.5*(a-c))**2 + b*b
    d = torch.sqrt(d2 + 1e-30)
    sinhc = torch.where(d < 1e-4, 1.0 + d2/6.0 + d2*d2/120.0, torch.sinh(d)/d)
    e = torch.exp(m); ch = torch.cosh(d)
    cxx = e*(ch + sinhc*0.5*(a-c)); cyy = e*(ch - sinhc*0.5*(a-c)); cxy = e*sinhc*b
    return cxx, cxy, cyy
```

Costo: un forward aggiuntivo trascurabile; beneficio: nella mia esperienza 1–1.5 ordini di grandezza sulla loss costitutiva a $\mathrm{Wi}\gtrsim 0.3$.

**(C) Ansatz di equilibrio locale (physics‑guided output layer) — la più aggressiva.**
Poiché $\mathbf M^{-1}$ è calcolabile analiticamente (3×3), imponete

$$
\mathbf t_\theta(x,y) = \underbrace{\mathbf M^{-1}_{\text{clamped}}\,\big[2\mu_p\,\mathbf d\big]}_{\text{soluzione locale omogenea esatta}} \;+\; \boldsymbol\tau_{\text{scale}}\odot \mathbf N_\theta(x,y),
$$

dove $\mathbf d=(D_{xx},D_{xy},D_{yy})$. Il primo termine cattura *esattamente* l'amplificazione $(1-\mathrm{De})^{-1}$ e tutta la dipendenza parametrica; la rete deve apprendere solo la **correzione advettiva** (memoria lungo le linee di corrente), che è liscia e $O(1)$. Attenzione: rende $\boldsymbol\tau$ dipendente da derivate di $\psi$, quindi la costitutiva richiede derivate seconde di $\psi$ (accettabile) e accoppia i gradienti di `model_tau` e `model_psi` (va bene in Fase 1, dove entrambi sono mobili).

---

## 2. Ancoraggio dello stress sui rulli: impatto e schedule di pesatura

### 2.1 Perché è indispensabile (e cosa identifica esattamente)

Come da §0, in Fase 1 il roll‑stress BC è la **sola** informazione su $(\lambda,\mu_p)$. Ma non tutte le componenti informano allo stesso modo. In prossimità della superficie di un rullo il flusso è quasi‑viscometrico (taglio semplice con rate $\dot\gamma$), e Oldroyd‑B dà:

$$
\tau_{xy}=\mu_p\dot\gamma,\qquad N_1=\tau_{xx}-\tau_{yy}=2\mu_p\lambda\dot\gamma^2,\qquad \tau_{yy}=0 .
$$

Da cui l'inversione esatta:

$$
\boxed{\ \mu_p=\frac{\tau_{xy}}{\dot\gamma},\qquad \lambda=\frac{N_1}{2\,\tau_{xy}\,\dot\gamma}=\frac{\mu_p N_1}{2\tau_{xy}^2}. }
$$

**Tutta l'informazione su $\lambda$ è contenuta nella prima differenza di sforzi normali $N_1$ sui rulli.** Se pesate le tre componenti "1:1" come fate ora (`W_ROLL_STRESS=1.0` uniforme) e le normalizzate con un unico `tau_scale`, il canale $\tau_{xy}$ (che è $O(\mu_p\dot\gamma)$) domina numericamente il canale $N_1$ (che è $O(\mu_p\lambda\dot\gamma^2)$, cioè un fattore $\mathrm{Wi}$ più piccolo se $\mathrm{Wi}<1$). **Risultato: il gradiente su $\mu_p$ è forte, quello su $\lambda$ è soppresso di un fattore $\mathrm{Wi}$ → plateau su $\lambda$.** Questa è, insieme al punto §0/§4, la spiegazione quantitativa della vostra Domanda 4.

**Correzione minima e ad altissimo impatto:**

```python
# normalizzazione per-componente sulla statistica dei target di bordo
s_xx = tau_bc[:,0].std(); s_xy = tau_bc[:,1].std(); s_yy = tau_bc[:,2].std()
# meglio ancora: riformulare in (N1, tau_xy, traccia)
loss_bc_tau = ( ((N1_pred - N1_tgt)/s_N1)**2 ).mean() * W_N1 \
            + ( ((txy_pred - txy_tgt)/s_xy)**2 ).mean() * W_SHEAR \
            + ( ((tr_pred  - tr_tgt )/s_tr)**2 ).mean() * W_TR
```
con `W_N1 ≈ 3–5 × W_SHEAR`. La base $(N_1,\tau_{xy},\mathrm{tr}\,\boldsymbol\tau)$ è quella che **diagonalizza approssimativamente la matrice di informazione di Fisher** rispetto a $(\lambda,\mu_p)$, eliminando la correlazione $\rho(\hat\lambda,\hat\mu_p)\approx -1$ che nella base $(\tau_{xx},\tau_{xy},\tau_{yy})$ produce la classica valle stretta nel landscape.

### 2.2 Rischi

| Rischio | Meccanismo | Mitigazione |
|---|---|---|
| **Sovradeterminazione caratteristica** | La costitutiva è iperbolica in $\boldsymbol\tau$: ammette dati solo sull'*inflow*. Sui rulli le linee di corrente sono chiuse (orbite periodiche) → la soluzione è determinata dalla monodromia $\oint$, non da Dirichlet. Imporre $\boldsymbol\tau$ ovunque sul bordo è formalmente sovradeterminato. | Non è fatale (i dati vengono da COMSOL, quindi *sono* compatibili) ma genera conflitto residuo⇄BC nei punti dove COMSOL è sotto‑risolto. Usare **loss robusta** (Huber, $\delta=2\sigma$) sui BC di stress. |
| **Assorbimento dell'errore di modello** | Con `W_BC_1=5` e rete $\boldsymbol\tau$ a 8×128, la rete può fittare i valori di bordo **e** pagare il residuo costitutivo solo in uno strato sottile spesso $O(10^{-3})$: la loss totale scende, $\lambda$ **non si muove**. È la patologia n.1 dei PINN inversi. | **Imporre il BC in forma *hard*** con ansatz a funzione distanza: $\boldsymbol\tau(x)=\boldsymbol\tau_{bc}(x)+\phi(x)\,\mathbf N_\theta(x)$, $\phi|_{\Gamma_{roll}}=0$. Così l'unica via per abbassare $\mathcal{R}_c$ è correggere $\lambda,\mu_p$. Con 4 cerchi: $\phi=\prod_i\tanh\!\big(( r_i^2-R^2)/\ell^2\big)$ (R‑function). |
| **Singolarità geometriche** | Nei punti di quasi‑contatto rullo/parete $\dot\gamma\to\infty$ e $\mathrm{De}_{loc}>1$: i dati COMSOL lì sono inaffidabili e $\boldsymbol\tau$ è quasi‑singolare. | Mascherare i punti BC con $2\lambda\sqrt{Q}>0.85$. |

### 2.3 Schedule di pesatura consigliata

Definisco $w_c$ (costitutiva), $w_b^{u}$ (no‑slip), $w_b^{\tau}$ (stress rulli), tutti **relativi**, con normalizzazione per‑componente già applicata.

| Fase (Adam FP32) | $w_c$ | $w_b^{\tau}$ | Parametri | Razionale |
|---|---|---|---|---|
| 0 – 2k | $0\to1$ (rampa lineare) | 10 | **congelati** | $\boldsymbol\tau\equiv0$ all'inizio ⇒ $\partial\mathcal L/\partial\lambda\equiv0$: aggiornare $\lambda$ ora significa iniettare rumore. Impostare `WARMUP_UNLOCK_EPOCH = 2000`. |
| 2k – 10k | 1 | 10 → 3 | attivi (VarPro ogni 100 it.) | Ampiezza di $\boldsymbol\tau$ imprintata; si comincia a far parlare la fisica interna. |
| 10k – 30k | 1 | 3 (NTK‑adattivo) | attivi | Bilanciamento automatico gradienti (Wang–Teng–Perdikaris): $w_i \leftarrow \frac{\sum_j \|\nabla_\theta \mathcal L_j\|}{\|\nabla_\theta\mathcal L_i\|}$, aggiornato ogni 500 it. con EMA $\alpha=0.9$. |
| 30k – 40k | 1 | 1 | attivi | *Annealing* finale: se $\lambda$ resta stabile mentre $w_b^\tau$ scende di 3×, la stima è **robusta**; se deriva, la stima era guidata dal bordo e va riportata l'incertezza. Questo è un test di consistenza, non solo una schedule. |

Aggiungete inoltre la **pesatura causale lungo le linee di corrente** (adattamento steady del *causal training* di Wang–Sankaran–Perdikaris): ordinate i punti di collocazione per tempo di residenza $s$ integrato lungo la traiettoria dall'inflow (o dal bordo) e pesate $w(s)=\exp(-\epsilon\sum_{s'<s}\mathcal R_c(s'))$. Poiché la costitutiva è un'ODE lungo le caratteristiche, imporre il residuo a valle prima che sia risolto a monte è esattamente il motivo per cui i PINN viscoelastici convergono a soluzioni non fisiche.

---

## 3. Transizione FP32 (Adam) → FP64 (L‑BFGS): rischi e best practice

### 3.1 Il problema *prima* di FP64: state disattivando 13 bit di mantissa

```python
torch.set_float32_matmul_precision("high")   # ⇐ ATTIVA TF32 SU AMPERE+
```

TF32 ha **10 bit di mantissa** ($\epsilon\approx 10^{-3}$). Il vostro residuo di momento richiede $\partial^3\psi$ (via $u_{xx}$, $u_{yy}$) e la curl‑momentum $\partial^4\psi$. Ogni ordine di differenziazione automatica propaga matmul in TF32: l'errore relativo sulle derivate terze è $O(10^{-3})$ **prima** di qualunque cancellazione catastrofica. In Fase 2 state cercando $\mu_s=0.1$ come coefficiente di $\nabla^2\mathbf u$ in una equazione dominata da $\nabla\cdot\boldsymbol\tau$ (con $\beta=0.1$, il termine solvente è il 10% del budget): **l'informazione su $\mu_s$ è sotto il rumore TF32**.

```python
torch.set_float32_matmul_precision("highest")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
```
Questa è, in termini di rapporto beneficio/costo, la singola riga più redditizia dell'intero progetto.

### 3.2 Rischi specifici del passaggio a FP64

1. **Il punto di partenza non è un punto critico della loss FP64.** Adam FP32 si arresta quando $\|\nabla\mathcal L\|\sim\eta_{\text{fp32}}$ (floor di rumore, tipicamente loss $\sim10^{-9}$–$10^{-10}$ per residui normalizzati). In FP64 quel gradiente rumoroso diventa un gradiente **deterministico ma di grande modulo relativo**: il primo passo di L‑BFGS (che senza coppie di curvatura usa $H_0=\gamma I$ con $\gamma=1$) può essere enorme. *Best practice*: dopo il cast, eseguire **200–500 step di Adam in FP64 con lr = 1e‑5** per "risedimentare" il punto, poi avviare L‑BFGS.
2. **Tolleranze di default che uccidono L‑BFGS.** `torch.optim.LBFGS` ha `tolerance_grad=1e-7` e `tolerance_change=1e-9`: in FP64, con loss $\sim10^{-10}$, l'ottimizzatore **esce alla prima iterazione**. È il bug più frequente in questa pipeline.
   ```python
   opt = torch.optim.LBFGS(params, lr=1.0, max_iter=25, history_size=100,
                           tolerance_grad=1e-16, tolerance_change=1e-16,
                           line_search_fn='strong_wolfe')
   for k in range(N_outer):          # N_outer * 25 = budget totale
       opt.step(closure)
   ```
   Chiamate esterne multiple con `max_iter=20–50` (invece di un unico `max_iter=10000`) permettono logging, early stopping e reset della memoria quasi‑Newton quando la curvatura è inconsistente.
3. **Determinismo della closure.** L‑BFGS costruisce $H_k$ da coppie $(s_k,y_k)$: se la closure resampla i punti di collocazione o usa chunking non deterministico, $y_k^{\mathsf T}s_k$ diventa rumoroso, la condizione di curvatura fallisce e l'update viene saltato (o peggio, accettato con curvatura negativa). Requisiti: (i) full‑batch fisso; (ii) se accumulate su chunk per la VRAM, i chunk devono essere **identici e nello stesso ordine** ad ogni iterazione, e la loss deve essere $\frac{1}{N}\sum_{\text{chunk}}\sum_{i}$, non media di medie (a meno che i chunk siano equidimensionali); (iii) `torch.use_deterministic_algorithms(True)`.
4. **Cast incompleto dello stato fisico.** `convert_to_fp64` deve promuovere anche i *buffer* (`eta_0`, `guess_lam`, `guess_mu_p`, `tau_scale`, `p_scale`, `H_ref`, `U_ref`) e i tensori dati. Un solo buffer FP32 forza un downcast silenzioso in un prodotto → si perde tutto il vantaggio senza errore. Verificate con un assert:
   ```python
   assert all(t.dtype==torch.float64 for t in
              list(model.parameters())+list(model.buffers())
              +list(physics.parameters())+list(physics.buffers()))
   ```
5. **Costanti di regolarizzazione tarate su FP32.** `inverse_softplus(min_val=1e-8)`, `torch.clamp(..., 20.0)`, `+1e-12` in `beta`: in FP64 introducono *kink* non differenziabili. Il line search di Wolfe su una funzione con derivata discontinua fallisce (`max_ls` esaurito) e L‑BFGS si blocca. Rendetele dipendenti dal dtype: `eps = torch.finfo(dtype).eps**0.5`.
6. **Ottimizzatore Adam non ricostruito.** Gli stati `exp_avg`/`exp_avg_sq` in FP32 castati a FP64 conservano il *bias* del rumore FP32. Ricostruire l'ottimizzatore da zero al cambio di precisione (e resettare `step`).
7. **Costo.** Su GPU consumer (GeForce) FP64 è 1/32–1/64 del throughput FP32. `LBFGS_MAX_ITERS_PHASE1 = 10000` in FP64 con 8×128×3 reti e derivate terze è dell'ordine di giorni. Strategia realistica: **FP64 solo per gli ultimi 500–1500 passi**, oppure precisione mista chirurgica (rete FP32, ma `x`, accumulo dei residui e loss in FP64 con `float64` sui soli tensori di residuo — riduce la cancellazione catastrofica nella somma senza pagare i matmul).

### 3.3 Test diagnostico per decidere *se* serve FP64

Prima di investire: valutate la stessa loss sullo stesso set di punti in FP32 e in FP64. Se
$$
\frac{|\mathcal L_{32}-\mathcal L_{64}|}{\mathcal L_{64}} < 10^{-2},
$$
il floor non è ancora numerico e il plateau ha origine nell'ottimizzazione o nel bilanciamento dei pesi: FP64 non vi aiuterà. Fatelo anche componente per componente (momento vs costitutiva): tipicamente il momento tocca il floor molto prima.

---

## 4. Parametrizzazione log‑space di $\lambda$: geometria del gradiente e come eliminare i plateau

### 4.1 Cosa fa realmente $\lambda=\lambda_g e^{r}$

$$
\frac{\partial\mathcal L}{\partial r}=\lambda\frac{\partial \mathcal L}{\partial\lambda},
\qquad \Delta\ln\lambda = -\eta_r\,\hat g_r .
$$

Pro: positività garantita, invarianza di scala, passi **moltiplicativi** (naturali per una quantità che varia su ordini di grandezza). Contro: la metrica indotta è $g_{rr}=\lambda^2 (\partial^2\mathcal L/\partial\lambda^2)$; il gradiente si annulla doppiamente dove $\mathcal L$ è piatta *e* $\lambda$ è piccolo.

**Punto chiave spesso frainteso:** con Adam il modulo del gradiente è irrilevante — il passo è $\approx\eta$ per costruzione ($\hat m/\sqrt{\hat v}\approx\pm1$). Quindi con `BASE_LR=1e-3`, `PARAM_LR_FACTOR=0.1` ⇒ $\eta_r=10^{-4}$ ⇒ **$\lambda$ può variare al massimo dello 0.01% per step**. Non è un plateau della loss: è un **limite di velocità imposto da voi**. Per passare da $\lambda=0.04$ a $0.05$ servono $\ln(1.25)/10^{-4}\approx 2230$ step *tutti nella stessa direzione* — e la direzione è corretta solo se il SNR del gradiente parametrico è alto. Con $\hat g_r$ dominato dal rumore, il moto è un random walk con deriva $\propto \mathrm{SNR}$ e vi servono $O(\mathrm{SNR}^{-2})$ volte più step.

### 4.2 Rimedi immediati (basso costo)

1. **Gruppo di parametri fisici separato** con:
   * $\eta_r = 5\times10^{-3}$ (passo dell'0.5% per step: 500 step per un movimento del 250%),
   * $\beta_1=0.99,\ \beta_2=0.999$ (EMA lungo ⇒ filtro passa‑basso sul rumore, aumenta l'SNR di $\sqrt{1/(1-\beta_1)}\approx 10$),
   * **`eps = 1e-12`** (non `1e-7`!). Con `ADAM_EPS=1e-7` e $\sqrt{\hat v}\sim10^{-8}$ (tipico per un parametro scalare a fine training), $\hat m/(\sqrt{\hat v}+\epsilon)\approx \hat m/\epsilon \ll 1$: **Adam smette letteralmente di aggiornare i parametri fisici**. Questa è, con altissima probabilità, la causa immediata del vostro plateau su $\lambda$.
2. **Trust region esplicita**: `r.data.clamp_(r_prev-0.02, r_prev+0.02)` per step, invece di `PARAM_CLIP_NORM` sulla norma del gradiente (che è mal definita per uno scalare).
3. **Barriera sul Weissenberg critico**: penalità $\mathcal L_{bar}=\zeta\,\mathrm{softplus}\!\big(2\lambda\sqrt{Q_{\max}}-0.9\big)^2$, per impedire il salto oltre il polo durante l'esplorazione.
4. **Riparametrizzazione decorrelante**: come mostrato in §2.1, i dati identificano $\mu_p$ (dal taglio) e $\lambda\mu_p$ (da $N_1$). Ottimizzate in $(\,r_1=\ln\mu_p,\ r_2=\ln(\lambda\mu_p)\,)$: la Hessiana diventa quasi‑diagonale e la valle stretta sparisce.

### 4.3 La soluzione definitiva: *variable projection* (Golub–Pereyra)

Poiché il residuo è **affine in $(\lambda,\mu_p)$** a $(\boldsymbol\tau,\mathbf u)$ fissati:

$$
\mathcal R_c = \boldsymbol\tau + \lambda\,\mathbf G[\boldsymbol\tau,\mathbf u] - 2\mu_p\mathbf D,
\qquad \mathbf G=\mathbf u\cdot\nabla\boldsymbol\tau-\mathbf L\boldsymbol\tau-\boldsymbol\tau\mathbf L^{\mathsf T},
$$

i parametri ottimi si ottengono in **forma chiusa** risolvendo un sistema 2×2:

$$
\begin{pmatrix}\langle \mathbf G,\mathbf G\rangle & -2\langle \mathbf G,\mathbf D\rangle\\[2pt]
-2\langle \mathbf G,\mathbf D\rangle & 4\langle \mathbf D,\mathbf D\rangle\end{pmatrix}
\begin{pmatrix}\lambda\\ \mu_p\end{pmatrix}
=\begin{pmatrix}-\langle \mathbf G,\boldsymbol\tau\rangle\\ 2\langle \mathbf D,\boldsymbol\tau\rangle\end{pmatrix},
$$
con $\langle\cdot,\cdot\rangle$ prodotto scalare pesato (usate gli stessi pesi di equilibrazione del §1.3A, e includete i punti di bordo con il peso $w_b^\tau$ per rompere la degenerazione $\mu_p\to0$).

```python
@torch.no_grad()
def varpro_update(G, D, tau, w):        # tensori (N,3), w pesi (N,1)
    a11 = (w*G*G).sum(); a12 = -2*(w*G*D).sum(); a22 = 4*(w*D*D).sum()
    b1  = -(w*G*tau).sum(); b2 = 2*(w*D*tau).sum()
    det = a11*a22 - a12*a12
    lam = ( a22*b1 - a12*b2)/det
    mup = (-a12*b1 + a11*b2)/det
    return lam.clamp(min=1e-4), mup.clamp(min=1e-3)
```

Applicatelo ogni 50–100 epoche in un blocco `no_grad` e ri‑settate `_raw_lam = log(lam/guess_lam)`. Vantaggi:
* elimina completamente il plateau (nessuna discesa del gradiente sui parametri);
* riduce la dimensione effettiva del problema non lineare (il funzionale ridotto $\tilde{\mathcal L}(\theta)=\mathcal L(\theta,\lambda^\star(\theta),\mu_p^\star(\theta))$ ha condizionamento migliore — risultato classico di Golub–Pereyra);
* fornisce **gratuitamente** la matrice di informazione (la 2×2 sopra) e quindi le barre d'errore su $\hat\lambda,\hat\mu_p$, oltre alla correlazione $\rho$: se $|\rho|>0.99$ sapete che la stima non è identificabile e dovete cambiare pesatura (§2.1).

**Estensione alla Fase 2 — $\mu_s$ in forma chiusa.** Osservazione fondamentale: nel momento
$$
\rho\,\mathbf u\cdot\nabla\mathbf u=-\nabla p+\mu_s\nabla^2\mathbf u+\nabla\cdot\boldsymbol\tau,
$$
il campo $p$ è **libero** e può assorbire qualsiasi componente irrotazionale del residuo. Per la decomposizione di Helmholtz, **$\mu_s$ è identificabile solo attraverso la parte solenoidale**, cioè l'equazione della vorticità:
$$
\rho\,\mathbf u\cdot\nabla\omega=\mu_s\nabla^2\omega+\underbrace{\partial_x(\nabla\!\cdot\!\boldsymbol\tau)_y-\partial_y(\nabla\!\cdot\!\boldsymbol\tau)_x}_{\;C(x,y)\ \text{(precalcolabile: }\boldsymbol\tau\text{ è congelato)}},\qquad \omega=-\nabla^2\psi .
$$
Con $\psi$ congelato per un istante, questa è **lineare in $\mu_s$** e la stima ottima è uno scalare:
$$
\boxed{\ \mu_s^\star=\frac{\big\langle \nabla^2\omega,\ \rho\,\mathbf u\cdot\nabla\omega-C\big\rangle}{\big\langle\nabla^2\omega,\nabla^2\omega\big\rangle}\ }
$$
Vedo che avete già una chiave `loss_curl` nel plotter: portatela al centro della Fase 2. Beneficio collaterale enorme: **potete identificare $\mu_s$ senza `model_p`**, eliminando 130k parametri di *nuisance* e la loss `W_BC_2` di ancoraggio della pressione dal problema inverso; $p$ si ricostruisce a posteriori risolvendo un Poisson $\nabla^2p=\nabla\cdot(\nabla\cdot\boldsymbol\tau-\rho\mathbf u\cdot\nabla\mathbf u)$ o per integrazione di linea. Costo: servono $\nabla^4\psi$ — introducete una testa ausiliaria $\omega_\theta$ con vincolo $\omega_\theta=-\nabla^2\psi$ (formulazione mista/first‑order) per abbassare l'ordine da 4 a 2, altrimenti in FP32 il rumore su $\nabla^4\psi$ è $O(1)$.

---

## 5. Le 5 modifiche a più alto impatto, in ordine di priorità

### #1 — Rendere identificabili i parametri e risolverli in forma chiusa (VarPro)
*Interventi*: (a) equilibrazione per‑componente dei residui e dei BC di stress (§1.3A, §2.1) con riformulazione in base $(N_1,\tau_{xy},\mathrm{tr}\,\boldsymbol\tau)$; (b) VarPro 2×2 per $(\lambda,\mu_p)$ ogni 100 epoche in Fase 1; (c) VarPro scalare per $\mu_s$ sull'equazione della vorticità in Fase 2; (d) `eps=1e-12` e `lr=5e-3` sul gruppo parametri se mantenete anche il gradiente.
*Impatto atteso*: errore su $\lambda$ da $O(10\%)$ a $O(1\%)$, e soprattutto **eliminazione del plateau**. Costo di implementazione: ~1 giorno.

### #2 — Formulazione log‑conformation (o, come minimo, precondizionamento $\mathbf M^{-1}$) per $\boldsymbol\tau$
Converte il polo $(1-2\lambda\dot\varepsilon)^{-1}$ in un logaritmo, garantisce $\mathbf c\succ0$, comprime il dynamic range del filamento birifrangente e disaccoppia i tre canali (§1.3B/C). È l'unica modifica che vi permetterà di salire in $\mathrm{Wi}$ senza riprogettare tutto.
*Impatto*: 1–1.5 ordini di grandezza sulla loss costitutiva; stabilità garantita.

### #3 — Igiene numerica: TF32 off, L‑BFGS configurato correttamente, audit dimensionale
(a) `allow_tf32=False` + `matmul_precision("highest")`; (b) `tolerance_grad/change = 1e-16`, `history_size=100`, `strong_wolfe`, chiamate a blocchi di 25 iterazioni; (c) 300 step Adam FP64 @1e‑5 prima di L‑BFGS; (d) closure deterministica full‑batch; (e) **audit dimensionale del residuo di momento**: verificate che `Re_scale*(u*u_x) + p_x - mu_s_nd*(u_xx+u_yy) - div_tau_x` sia consistente — attualmente $u$ e $p$ sembrano dimensionali mentre $\mathrm{Re}$ e $\mu_s^*$ sono adimensionali; se la normalizzazione non è esattamente $\hat u=u/U_{ref},\ \hat x=x/H_{ref},\ \hat p=pH/(\eta_0U)$, i termini hanno pesi relativi sbagliati di ordini di grandezza. Un test di verifica: sostituite la soluzione COMSOL interpolata nel residuo — deve dare $\sim10^{-3}$ relativo, non $10^{+2}$.
*Impatto*: abbassa il floor numerico di 2–3 ordini; rende utile la Fase 2.

### #4 — Simmetria $D_4$ imposta in forma hard + collocazione adattiva guidata dall'Okubo–Weiss
Se la configurazione COMSOL è simmetrica (4 rulli identici, contro‑rotanti), valgono esattamente
$$
\psi(-x,y)=-\psi(x,y),\quad \psi(x,-y)=-\psi(x,y),\quad \psi(y,x)=\psi(x,y).
$$
Realizzazione esatta con $\psi_\theta = x\,y\;\mathcal N_\theta(x^2+y^2,\ x^2y^2)$: **il punto di stagnazione è inchiodato analiticamente all'origine** (errore zero sulla posizione, che è la quantità più sensibile per $\dot\varepsilon$ locale e quindi per $\lambda$), il dominio effettivo si riduce a 1/8 e i gradi di libertà spuri crollano. Per $\boldsymbol\tau$: $\tau_{xx}(y,x)=\tau_{yy}(x,y)$, $\tau_{xy}(y,x)=-\tau_{xy}(x,y)$ (verificate i segni sui dati prima di imporre). In parallelo: resampling RAD/RAR con densità $\propto \mathcal R_c^k$ **più** un termine $\propto\max(Q,0)$, per infittire su filamento e separatrici.
*Impatto*: riduzione tipica dell'errore $L^2$ su $u,v$ di 3–5×; convergenza 2× più rapida.

### #5 — Bilanciamento automatico dei pesi (NTK/grad‑norm) + pesatura causale lungo le caratteristiche
Sostituire `W_DATA_1/W_BC_1/W_CONSTITUTIVE/W_DATA_2/...` (7 costanti tarate a mano) con:
$$
w_i^{(k+1)}=(1-\alpha)w_i^{(k)}+\alpha\frac{\sum_j\|\nabla_\theta\mathcal L_j\|_2}{n\,\|\nabla_\theta\mathcal L_i\|_2},\qquad\alpha=0.1,
$$
aggiornato ogni 500 iterazioni, **più** moltiplicatori self‑adaptive puntuali (SA‑PINN) sui BC di stress, che down‑pesano automaticamente i punti incompatibili vicino alle singolarità geometriche, **più** la pesatura causale in ascissa curvilinea lungo le linee di corrente per rispettare l'iperbolicità della costitutiva (§1.2, §2.3).
*Impatto*: rimuove il fattore umano dominante nella varianza run‑to‑run e previene l'assorbimento dell'errore di modello.

---

## Appendice — checklist diagnostica da inserire nel logging TensorBoard

| Metrica | Perché |
|---|---|
| $\max_\Omega 2\lambda\sqrt{\max(Q,0)}$ | Distanza dalla catastrofe coil–stretch; deve restare $<0.9$. |
| $\kappa$ della matrice VarPro 2×2 e $\rho(\hat\lambda,\hat\mu_p)$ | Identificabilità in tempo reale. $|\rho|>0.99 \Rightarrow$ ripesare $N_1$. |
| $\|\nabla_\theta\mathcal L_i\|$ per ogni termine di loss | Diagnosi del gradiente dominante (di solito i BC schiacciano la PDE di 2–3 ordini). |
| $\mathcal L_{32}$ vs $\mathcal L_{64}$ sugli stessi punti | Decide se il plateau è numerico o di ottimizzazione. |
| $\min_\Omega \lambda_{\min}(\mathbf c)$ | Positività della conformazione: se $<0$ la soluzione è non fisica anche con loss bassa. |
| $\|\nabla\cdot\mathbf u\|$ | Dev'essere $\sim\epsilon_{macchina}$ (formulazione stream‑function): se non lo è, c'è un bug nel fattore $H_{ref}/H_{coord}$ di `_grad`. |

Se dovessi indicare **una sola** azione da compiere oggi: cambiare `ADAM_EPS` per il gruppo dei parametri fisici da `1e-7` a `1e-12` e disattivare TF32; poi implementare il VarPro. Le prime due sono modifiche di due righe che, quasi certamente, sbloccano il plateau su $\lambda$ che state osservando.