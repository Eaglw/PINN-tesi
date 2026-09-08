# Analisi Convergenza PINN Viscoelastico (final_roll)

**Modello:** `anthropic/claude-sonnet-5@default`  

---

# Analisi Critica del Solver ViscoelasticNet-PINN per Four-Roll Mill

---

## 1. Stiffness delle PDE e Bilanciamento delle Loss

### 1.1 Origine fisico-matematica della stiffness

Il residuo costitutivo di Oldroyd-B implementato (derivata upper-convected) è:

$$
\boldsymbol{\tau} + Wi\left(\mathbf{u}\cdot\nabla\boldsymbol{\tau} - (\nabla\mathbf{u})^T\cdot\boldsymbol{\tau} - \boldsymbol{\tau}\cdot\nabla\mathbf{u}\right) = 2\mu_p^{*} \mathbf{D}
$$

Nel codice, per la componente $\tau_{xx}$:

```python
upper_xx = u * tau_xx_x + v * tau_xx_y - 2*u_x*tau_xx - 2*u_y*tau_xy
```

Il termine $-2u_x\tau_{xx}$ è **bilineare** nello stato ($\tau_{xx}$ moltiplicato per il gradiente di velocità locale $u_x = \dot{\varepsilon}$). Questo è *fisicamente* il meccanismo di stretching molecolare, ma numericamente introduce un comportamento simile a un'ODE lineare con guadagno variabile nello spazio:

$$
\frac{d\tau_{xx}}{dt}\bigg|_{\text{lungo la traiettoria}} \sim \frac{1}{\lambda}\left(2\lambda\dot\varepsilon(x) - 1\right)\tau_{xx} + \text{sorgente}
$$

Lungo l'asse di uscita del punto di stagnazione iperbolico, $\dot\varepsilon(x) = u_x$ è massimo. La soluzione analitica stazionaria per flusso estensionale puro di Oldroyd-B è nota:

$$
\tau_{xx} = \frac{2\mu_p\dot\varepsilon}{1 - 2\lambda\dot\varepsilon}, \qquad \text{divergente per } Wi_{loc} = \lambda\dot\varepsilon \to \tfrac{1}{2}
$$

**Questo è il vero cuore della stiffness**: non è un artefatto numerico ma la **singolarità del coil-stretch/Weissenberg critico** ($Wi_{crit}=0.5$) intrinseca al modello Oldroyd-B in flusso estensionale. Se durante l'ottimizzazione $\lambda$ (parametro inverso, libero) si avvicina a un valore per cui $Wi_{loc}(x_{\text{centro}}) \to 0.5$, il residuo costitutivo diventa mal condizionato *localmente* — gradiente rispetto ai pesi di `model_tau` esplode in un intorno del punto di stagnazione, mentre resta regolare altrove. È uno stiffness **spazialmente eterogeneo**, non uniforme, quindi tecniche di normalizzazione globale (come l'attuale divisione per `tau_scale` scalare) non lo risolvono.

Il residuo di momento, al contrario, è governato da operatori lineari ellittici ($\mu_s^*\nabla^2\mathbf{u}$, $\nabla p$) più il termine di div$(\boldsymbol{\tau})$ (sorgente, non feedback): è quindi molto meglio condizionato, ma **eredita** la stiffness di $\boldsymbol{\tau}$ tramite `div_tau_x`, `div_tau_y`. Questo giustifica pienamente la scelta architetturale di **congelare `model_tau` in Fase 2**: se non fosse congelato, il feedback $\tau\to$momento$\to\tau$ (via backprop condiviso) amplificherebbe l'instabilità nella regione critica.

### 1.2 Gradient imbalance tra componenti di stress

Nel four-roll mill, l'anisotropia delle componenti è drammatica:
- $\tau_{xx}$ cresce come $O(1/(1-2Wi_{loc}))$ vicino al centro/outflow — può essere 1-2 ordini di grandezza superiore a $\tau_{xy}, \tau_{yy}$.
- $\tau_{xy}$ domina invece vicino alle pareti dei rulli (shear-dominated).

La loss attuale:
```python
loss_c = (f_txx**2 + f_tyy**2 + f_txy**2).mean() / 3.0
```
tratta le tre componenti in modo isotropo dopo la normalizzazione con un **unico** `tau_scale` scalare. Questo produce un mismatch: se `tau_scale` è calibrato sul valore RMS globale (dominato da $\tau_{xx}$ nella zona estensionale), allora il residuo $f_{txy}$ (che opera su valori molto più piccoli) viene numericamente "silenziato" nella loss aggregata — la rete privilegia la correttezza di $\tau_{xx}$ a scapito di $\tau_{xy}$, con effetto a cascata sul termine di shear $\mu_p^*(u_y+v_x)$ nella BC di no-slip.

**Raccomandazione matematica**: sostituire lo scalare `tau_scale` con normalizzazione per-componente:

$$
\tilde f_{\tau_{ij}} = \frac{f_{\tau_{ij}}}{s_{ij}}, \qquad s_{ij} = \text{RMS}(\tau_{ij})_{\text{dati/checkpoint}}
$$

oppure, meglio ancora, adottare un **residual-based attention weighting** (Wang et al. 2022, McClenny & Braga-Neto SA-PINN) dove i pesi $\lambda(x)$ sono funzioni apprendibili/adattive che si concentrano automaticamente dove il residuo è maggiore — questo risolverebbe *sia* l'eterogeneità tra componenti *sia* quella spaziale (centro vs. rulli) in un colpo solo.

---

## 2. Ancoraggio dello Stress sui Rulli (`USE_ROLL_STRESS_BC`)

### 2.1 Sufficienza fisica del vincolo

Sui rulli il flusso locale è dominato da shear (no-slip + trascinamento tangenziale), un regime cinematico **qualitativamente diverso** da quello iperbolico del centro. L'equazione costitutiva in shear semplice ha soluzione:

$$
\tau_{xy} = \mu_p^*\dot\gamma, \quad \tau_{xx} = 2\lambda\mu_p^*\dot\gamma^2, \quad \tau_{yy}=0
$$

quindi l'ancoraggio sui rulli fornisce informazione forte e ben condizionata su $\tau_{xy}$ (lineare in $\dot\gamma$) ma informazione debole e di ordine superiore su $\tau_{xx}$ (quadratica, piccola se $\dot\gamma$ moderato). **Non c'è alcuna BC che vincoli direttamente il regime estensionale del centro**: quella regione è "supervisionata" solo indirettamente tramite:
1. Il residuo PDE costitutivo stesso (che è proprio la zona più stiff, punto 1);
2. Il termine di produzione $-2\mu_p^*u_x$ nell'equazione di $\tau_{xx}$, che dipende dai dati di velocità ($u,v$) via `model_psi`.

Questo crea un rischio concreto: **la rete può soddisfare quasi esattamente le BC sui rulli (overfitting locale a bassa dimensionalità del bordo) mentre nell'interno — specialmente vicino al punto di stagnazione — il campo $\tau_{xx}$ è sotto-vincolato**, con soluzioni multiple compatibili con il residuo PDE a bassa loss ma fisicamente errate (tipico degli operatori mal posti quando manca supervisione diretta nell'interno). Questo è aggravato dal fatto che la loss è mediata su tutto il dominio: pochi punti vicino al centro rispetto alla massa di punti "facili" lontano da esso.

### 2.2 Raccomandazione sul peso $W_{roll\_stress}$

Non consiglio un rapporto 1:1 statico come nell'attuale default. Ragioni:

- **All'inizio del training** `model_tau` è inizializzato con last-layer zero (`initialize_last_layer_zero`), quindi $\boldsymbol{\tau}\approx 0$ ovunque: la BC sui rulli (se target $\ne 0$) genera gradienti enormi e rumorosi contro una rete ancora non informata cineticamente, rischiando di "ancorare" male la soluzione prima che $\psi$ e il residuo costitutivo abbiano stabilizzato la forma globale del campo.
- Consiglio uno **schema ad annealing**:
$$
W_{roll\_stress}(epoch) = W_{roll\_stress}^{max}\cdot\min\!\left(1,\ \frac{epoch}{\tau_{warmup}}\right), \quad W_{roll\_stress}^{max}\in[0.3, 1.0]
$$
partendo basso ($\sim0.1$–$0.2$) e salendo a $\sim0.5$–$1.0$ dopo qualche migliaio di epoche Adam, quando il residuo costitutivo ha già una forma ragionevole.
- **Fondamentale**: aggiungere qualche punto di collocazione soft-supervisionato *vicino al centro* (non solo sui rulli), anche solo con un vincolo debole $\tau_{yy}\approx -\tau_{xx}$ (traccia nulla approssimata, valida se $\mu_s\ll\mu_p$... in realtà per Oldroyd-B in estensione pura $\tau_{yy}=-2\lambda\mu_p^*\dot\varepsilon/(1+2\lambda\dot\varepsilon)$, quindi non a traccia nulla esatta, ma il punto è che una debole regolarizzazione fisica locale nel bulk aiuta enormemente a disambiguare) o un termine di smoothness $\|\nabla\tau\|^2$ localizzato, per mitigare l'overfitting ai soli bordi.

---

## 3. Transizione Adam (FP32/TF32) → L-BFGS (FP64)

### 3.1 Insidie del cambio di precisione

1. **Buffer non convertiti**: `guess_lam`, `guess_mu_p`, `guess_mu_s`, `eta_0` sono registrati come `buffer` float32 (`register_buffer(..., dtype=torch.float32)`). Se `convert_to_fp64` (in `utils.py`, non mostrato per intero) opera solo su `model.parameters()` e non ricorsivamente su tutti i buffer di `physics`, si genera un **mismatch di dtype silente**: PyTorch farà upcasting implicito in alcune operazioni (es. `guess_lam * torch.exp(self._raw_lam)`), ma questo introduce un errore di troncamento residuo fp32 che **vanifica parzialmente il vantaggio della fase FP64**, specialmente per parametri come $\lambda$ dove la precisione fine è l'obiettivo esplicito della Fase L-BFGS. Verificare che `.double()` sia chiamato su **tutto** `physics` (moduli + buffer), non solo `model`.

2. **TF32 residuo**: `torch.set_float32_matmul_precision("high")` è impostato globalmente all'avvio e **non viene mai disabilitato** prima della fase L-BFGS. Anche convertendo pesi e dati a `float64`, se qualche operazione intermedia rimane in fp32 (es. costanti hardcoded, `RHO` in `physics.py` letto da `globals()` come float python — non è un problema di per sé, ma va verificato il dtype coerente in ogni prodotto), l'obiettivo di "alta precisione" della fase L-BFGS viene parzialmente eroso. Raccomando esplicitamente `torch.backends.cuda.matmul.allow_tf32 = False` e `torch.backends.cudnn.allow_tf32 = False` prima di avviare L-BFGS.

3. **Discontinuità della traiettoria di ottimizzazione**: il cast fp32→fp64 dei pesi è esatto (nessuna perdita, va solo in una direzione di precisione crescente), ma il valore della loss e dei suoi gradienti *ricalcolati* in fp64 può differire dal valore fp32 dell'ultimo step Adam per via del rumore TF32 pregresso (errore relativo $\sim 10^{-3}$ tipico del formato TF32, 10 bit di mantissa). Questo genera un piccolo "salto" nella loss all'avvio di L-BFGS che è normale e atteso, ma va monitorato per distinguere da un bug (se il salto è >1 ordine di grandezza, è indice di bug, non di rumore TF32).

4. **`GRAD_CLIP_NORM = 1000.0`**: è di fatto un clipping disattivato (i gradienti raramente raggiungono globalmente questa norma anche in condizioni instabili tipiche di reti profonde a 8×128). Non protegge dalle esplosioni locali dei gradienti di $\tau_{xx}$ vicino al centro (punto 1). Raccomando un clipping molto più stringente e *dinamico* (es. basato su percentile mobile della norma storica, o clip a $O(1$–$10)$) specialmente in Fase 1 Adam quando `model_tau` è ancora instabile.

### 3.2 Configurazione L-BFGS

Il codice non mostra esplicitamente `history_size` né `tolerance_grad`/`tolerance_change` per `torch.optim.LBFGS` — presumibilmente lasciati a default (`history_size=100`, `tolerance_grad=1e-7`, `tolerance_change=1e-9`). Su una superficie di loss fortemente non uniforme in curvatura (quasi-singolare vicino al centro iperbolico, quasi piatta lontano), ciò comporta rischi concreti:

- **Line search fallita / step nullo**: `strong_wolfe` richiede che sia soddisfatta la condizione di curvatura; se l'Hessiana approssimata (BFGS rank-2, costruita su `history_size` coppie $(s_k,y_k)$) diventa indefinita/mal condizionata a causa della componente quasi-singolare del residuo costitutivo, la line search può non trovare $\alpha>0$ soddisfacente e l'ottimizzatore si blocca in un plateau apparente (loss stagnante per migliaia di "iterazioni interne" senza progresso reale).
- Con `max_iter=10000` in **una singola chiamata** `.step(closure)` (tipico pattern PyTorch per LBFGS "batch" non stocastico), se si verifica uno stallo a metà, si perdono migliaia di iterazioni-budget senza possibilità di intervento (nessun logging intermedio se `closure` non stampa autonomamente).

**Raccomandazioni**:
- Impostare `history_size` esplicitamente e moderato (30–50) per limitare l'influenza di curvature "vecchie" non più rappresentative quando il punto operativo si sposta rapidamente.
- Suddividere il budget L-BFGS in **blocchi con restart** (es. 10 blocchi da 1000 iter, reinizializzando l'ottimizzatore tra un blocco e l'altro) invece di un'unica chiamata monolitica: questo permette (a) logging intermedio, (b) reset della history BFGS se degenera, (c) eventuale re-injection di rumore/perturbazione se la loss satura.
- Impostare `tolerance_grad`/`tolerance_change` più stringenti solo dopo aver verificato che il residuo costitutivo non sia strutturalmente non-liscio nella zona critica (altrimenti L-BFGS "crede" erroneamente di aver convergere per `tolerance_change` raggiunta quando in realtà è bloccato da una line search fallita).

---

## 4. Parametrizzazione Log-Space di $\lambda, \mu_p, \mu_s$

Con $\lambda = \lambda_{guess}\cdot e^{r_\lambda}$, si ha:

$$
\frac{\partial \mathcal{L}}{\partial r_\lambda} = \frac{\partial\mathcal{L}}{\partial\lambda}\cdot\frac{\partial\lambda}{\partial r_\lambda} = \frac{\partial\mathcal{L}}{\partial\lambda}\cdot\lambda
$$

### 4.1 Effetto su Adam (FP32)
Adam normalizza l'aggiornamento tramite la stima del secondo momento: $\Delta r_\lambda \approx -\eta\cdot\frac{\hat m}{\sqrt{\hat v}+\epsilon}$, quantità **adimensionale e invariante di scala** rispetto al gradiente stesso (a meno del rumore additivo). Questo significa che, *in assenza di rumore*, la parametrizzazione log-space non penalizza né avvantaggia la velocità di convergenza quando $\lambda\ll1$: Adam "cancella" lo scaling moltiplicativo per $\lambda$.

**Tuttavia**, il **rapporto segnale/rumore** peggiora: se il rumore sul gradiente $\partial\mathcal{L}/\partial\lambda$ ha componente additiva $\sigma_{noise}$ indipendente da $\lambda$ (dominato da TF32/batch noise), allora:

$$
\text{SNR}(r_\lambda) = \frac{\lambda\cdot|\partial\mathcal{L}/\partial\lambda|_{signal}}{\lambda\cdot\sigma_{noise}} = \text{SNR}(\lambda) \quad \text{(invariante)}
$$

— in realtà il rapporto S/N **non cambia** per il gradiente puro, ma cambia l'accumulo nella stima $\hat v$ di Adam se il rumore floating-point (non gaussiano ma quantizzazione TF32) ha una componente *assoluta fissa* $\epsilon_{fp}$ indipendente dallo scaling: in tal caso $\partial\mathcal{L}/\partial r_\lambda$ ha un pavimento di rumore *moltiplicato* per $\lambda$ piccolo, quindi quando $\lambda\to0$ il segnale utile si comprime relativamente al rumore floating-point assoluto (che non scala) — qui sì che si ha una **perdita di SNR proporzionale a $\lambda$**. Questo giustifica pienamente la necessità del refinement L-BFGS-FP64: passando a fp64, il pavimento di rumore floating-point si abbassa di ~8 ordini di grandezza, recuperando SNR sufficiente a risolvere $\lambda$ con precisione fine anche quando piccolo.

### 4.2 Effetto su L-BFGS (gradiente "raw", non normalizzato)
Qui lo scaling per $\lambda$ **conta esplicitamente**: L-BFGS costruisce l'approssimazione della Hessiana su gradienti raw. Se $\lambda_{true}=0.05$ (piccolo rispetto a $O(1)$), il gradiente $\partial\mathcal{L}/\partial r_\lambda$ sarà strutturalmente più piccolo di quello su $r_{\mu_p}$ (se $\mu_p\sim O(1)$), causando un **condizionamento asimmetrico dell'Hessiana approssimata multi-parametro**. Con un solo scalare per parametro il rischio è contenuto, ma se in futuro si aggiungono $\varepsilon_{PTT}$ o $\alpha_{Giesekus}$ come parametri liberi con guess diversi in scala, la disomogeneità tra gradienti raw può rallentare drasticamente L-BFGS su alcuni parametri rispetto ad altri.

**Raccomandazione**: monitorare separatamente la norma del gradiente per ciascun `_raw_*` durante L-BFGS, e se necessario introdurre un **precondizionamento diagonale esplicito** (riscalare artificialmente $r_\lambda, r_{\mu_p}, r_{\mu_s}$ per portarli a gradiente comparabile), oppure usare un ottimizzatore quasi-Newton con precondizionatore diagonale adattivo per i soli parametri fisici, disaccoppiato da quello usato sui pesi di rete.

---

## 5. Modifiche Pratiche Raccomandate (ordine di priorità)

1. **[Priorità Massima] Continuation method su $Wi$/$\lambda$ in Fase 1.**
   Introdurre un warm-up graduale sul tempo di rilassamento effettivo usato nel residuo costitutivo (es. $\lambda_{eff}(epoch) = \lambda_{guess}\to\lambda_{corrente}$ scalato con un fattore di rampa $0\to1$ nelle prime N epoche), o equivalentemente iniziare il training a $Wi$ ridotto artificialmente e rilassare il vincolo progressivamente. Questo è lo standard nei solver CFD per fluidi viscoelastici (continuazione in $Wi$) per evitare di "cadere" fin da subito nella regione super-critica ($Wi_{loc}>0.5$) dove la soluzione stazionaria diverge — rischio concreto qui dato che $\lambda$ è **libero** e potrebbe temporaneamente sovrastimare il vero valore durante l'esplorazione Adam.

2. **[Alta Priorità] Normalizzazione per-componente e/o pesi adattivi sul residuo costitutivo.**
   Sostituire lo scalare unico `tau_scale` con normalizzazione per componente ($\tau_{xx}, \tau_{xy}, \tau_{yy}$ separate) calcolata da percentili robusti (es. 95° percentile) dei dati/checkpoint, e valutare l'introduzione di residual-based attention weights spazialmente variabili per compensare l'eterogeneità centro/rulli.

3. **[Alta Priorità] Resampling adattivo dei punti di collocazione (RAR).**
   Infittire dinamicamente i punti di collocazione nella regione ad alto residuo (centro iperbolico e outflow axis), dove attualmente il campionamento è presumibilmente quasi-uniforme, per fornire più "peso statistico" alla zona fisicamente più critica senza dover alterare i pesi globali della loss.

4. **[Media Priorità] Gestione rigorosa della precisione nella transizione di fase.**
   Garantire cast esplicito a `float64` di **tutti** i buffer di `Physics` (non solo `model`), disabilitare TF32 (`torch.backends.cuda.matmul.allow_tf32=False`) prima di L-BFGS, e ridurre `GRAD_CLIP_NORM` a un valore realmente efficace (es. 5–20) durante Adam Fase 1 per contenere le esplosioni locali di gradiente su $\tau_{xx}$.

5. **[Media Priorità] L-BFGS a blocchi con restart e diagnostica di line-search.**
   Sostituire l'unica chiamata monolitica (`max_iter=10000`) con blocchi ripetuti (es. 10×1000) con reset della history BFGS e logging tra un blocco e l'altro, per rilevare tempestivamente stalli da line-search fallita ed eventualmente reagire (riduzione lr, perturbazione, cambio ordine parametri) prima di esaurire l'intero budget computazionale su un plateau spurio.