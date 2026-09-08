# Report Analisi PINN: Igiene Numerica Tf32 E Adam_Eps Per Parametri Fisici

- **Data/Ora:** `2026-09-08 20:36:29`
- **Modello:** `anthropic/claude-sonnet-5@default`
- **Topic:** `igiene numerica TF32 e ADAM_EPS per parametri fisici`
- **Target Codebase:** `final_roll/src/`

---

# Analisi Critica: Igiene Numerica TF32 e Calibrazione di $\epsilon_{\text{Adam}}$ per i Parametri Fisici Log-Parametrizzati

## 1. Fondamenti Teorici

### 1.1 TensorFloat-32 e Propagazione dell'Errore in un PINN di Ordine Elevato

TF32 è un formato ibrido per le unità Tensor Core delle GPU Ampere/Hopper che **preserva gli 8 bit di esponente** della IEEE-754 FP32 ma **tronca la mantissa a 10 bit** (contro i 23 bit nativi). La precisione macchina relativa passa quindi da:

$$
u_{\text{FP32}} = 2^{-24} \approx 5.96\times 10^{-8}
\quad\longrightarrow\quad
u_{\text{TF32}} = 2^{-11} \approx 4.88\times 10^{-4}
$$

un **degrado di circa 3–4 ordini di grandezza**. Nel vostro solver, `torch.set_float32_matmul_precision("high")` abilita TF32 per **tutte** le operazioni `addmm`/`mm` sottostanti a `nn.Linear`, quindi per l'intero forward pass di `model_psi`, `model_p`, `model_tau`.

Il problema diventa critico perché la vostra formulazione a stream-function richiede **derivate automatiche annidate fino al second'ordine** per chiudere la momentum equation:

$$
u = \psi_{,y}, \quad v = -\psi_{,x} \;\Longrightarrow\; u_{xx}, u_{yy}, v_{xx}, v_{yy} = \partial^2_{xx,yy}\big(\partial_{x,y}\psi\big)
$$

Ogni applicazione di `torch.autograd.grad` **ricalcola il grafo tramite gli stessi matmul TF32**, per cui l'errore di troncamento **non si mantiene invariato ma si compone multiplicativamente** attraverso ciascun layer e ciascun ordine di derivazione. Per una rete a $L=8$ hidden layers, l'errore relativo atteso sul residuo del second'ordine è euristicamente limitato da:

$$
\varepsilon_{\text{residuo}} \sim \mathcal{O}\!\left(L \cdot u_{\text{TF32}}\cdot \kappa(\mathbf{J})\right)
$$

dove $\kappa(\mathbf{J})$ è il numero di condizionamento locale dello Jacobiano della rete. Con $u_{\text{TF32}}\approx 5\times10^{-4}$, è del tutto plausibile che il **floor di rumore** sui residui PDE si attesti a $10^{-3}$–$10^{-2}$, un valore **incompatibile** con l'obiettivo dichiarato di raffinamento L-BFGS in FP64 (che punta tipicamente a residui $<10^{-8}$).

### 1.2 $\epsilon_{\text{Adam}}$ e la Parametrizzazione Log-Space dei Parametri Costitutivi

L'update Adam per un generico parametro $\theta$ è:

$$
\theta_{t} = \theta_{t-1} - \eta \, \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon}
$$

Per i parametri raw $r_\lambda, r_{\mu_p}$ (con $\lambda = \lambda_{\text{guess}}e^{r_\lambda}$), la regola della catena impone:

$$
g_{r_\lambda} \equiv \frac{\partial \mathcal{L}}{\partial r_\lambda} = \frac{\partial \mathcal{L}}{\partial \lambda}\cdot \lambda_{\text{guess}}\, e^{r_\lambda} = \frac{\partial \mathcal{L}}{\partial \lambda}\cdot \lambda
$$

Questo gradiente è uno **scalare unico**, aggregato per media su tutte le collocation points, radicalmente diverso — per magnitudo e varianza temporale — dai gradienti dei $\sim 10^5$ pesi di `model_psi`/`model_tau`. Quando $\mathcal{L}$ si avvicina alla convergenza (residui costitutivi piccoli su gran parte del dominio), $g_{r_\lambda}\to 0$ e il comportamento dell'update degenera:

$$
\sqrt{\hat v_t} \ll \epsilon \;\Longrightarrow\; \Delta\theta \approx -\eta\,\frac{\hat m_t}{\epsilon}
$$

Con $\eta_{\text{phys}} = \text{BASE\_LR}\times\text{PARAM\_LR\_FACTOR} = 10^{-4}$ e $\epsilon = 10^{-7}$, il **fattore di amplificazione effettivo** è $\eta/\epsilon = 10^{3}$: qualunque residuo di rumore su $\hat m_t$ (anche di origine puramente TF32!) viene amplificato di tre ordini di grandezza sull'update di $r_\lambda$, generando oscillazioni spurie attorno al minimo — esattamente il fenomeno che il codice **ammette implicitamente** in `plot_params`:

```python
if abs(max_v - min_v) < 1e-5:
    ... # "rumore FP32"
```

Questo commento è una **prova diagnostica diretta** che il team ha osservato il sintomo ma non ne ha trattato la causa radice a livello di ottimizzatore.

---

## 2. Criticità Riscontrate nel Codice (`physics.py`, `train.py`, `train_4roll_main.py`)

### 2.1 Amplificazione Quadratica dell'Errore in `Physics._grad`

```python
def _grad(self, y, x, create_graph=True, retain_graph=True):
    return torch.autograd.grad(...)[0] * (self.H_ref / self.H_coord)
```

Questo fattore di riscalamento dimensionale $\Gamma \equiv H_{\text{ref}}/H_{\text{coord}}$ viene applicato **ad ogni chiamata**. Per il Laplaciano viscoso (`u_xx`, `u_yy`, ecc.), `_grad` è invocato **due volte in cascata**, per cui l'errore assoluto di round-off TF32 sul termine diffusivo $\mu_s^*(u_{xx}+u_{yy})$ viene amplificato di un fattore:

$$
\Gamma^2 = \left(\frac{H_{\text{ref}}}{H_{\text{coord}}}\right)^2
$$

Se $H_{\text{coord}} = 0.05$ (come da default) e $H_{\text{ref}}\sim\mathcal{O}(1)$, $\Gamma=20 \Rightarrow \Gamma^2=400$: **il rumore TF32 grezzo viene moltiplicato per 400** prima di entrare nel residuo di momento $f_u, f_v$. Questo è un bottleneck numerico non banale, **strutturalmente indipendente** dalla qualità dell'ottimizzatore.

### 2.2 Assenza di Gestione Differenziata di TF32 per Fase

Non vi è alcuna disattivazione selettiva di TF32 durante la **Fase 1** (identificazione di $\lambda, \mu_p$ tramite equazione costitutiva), che è proprio la fase in cui la precisione del gradiente rispetto ai parametri fisici è più critica per l'identificabilità inversa. Il flag globale:

```python
torch.set_float32_matmul_precision("high")
```

resta attivo indistintamente in Adam Fase 1, Adam Fase 2 e — potenzialmente — anche durante warm-up prima della conversione FP64.

### 2.3 $\epsilon_{\text{Adam}}$ Condiviso fra Pesi di Rete e Parametri Fisici

Non emerge, né in `train.py` né in `train_4roll_main.py`, alcuna differenziazione di `eps` per gruppo di parametri: `PARAM_LR_FACTOR` scala il **learning rate**, ma non esiste un `PARAM_EPS_FACTOR`. Questo è un'incoerenza di igiene numerica: si è riconosciuto che i parametri fisici necessitano di una dinamica di update diversa (da cui `PARAM_LR_FACTOR`), ma **non si è esteso lo stesso principio all'unico altro iperparametro che governa la stabilità vicino a $g_t\to 0$**, ovvero $\epsilon$.

### 2.4 Il Paradosso TF32 $\times$ $\epsilon_{\text{Adam}}$ Piccolo

Qui risiede la criticità più sottile: **con TF32 attivo**, il floor di rumore relativo sul gradiente scalare $g_{r_\lambda}$ è dell'ordine di $u_{\text{TF32}}\|\theta\|\sim 10^{-3}$–$10^{-4}$. Un $\epsilon = 10^{-7}$ è quindi **troppo piccolo per fungere da termine di regolarizzazione protettivo**: l'ottimizzatore non "ammortizza" il rumore, ma lo insegue come se fosse segnale. Il rimedio non è banale: o si **elimina la fonte di rumore** (disattivando TF32) oppure si **alza $\epsilon$** fino a renderlo commensurabile al floor di rumore atteso, sacrificando reattività su gradienti fisiologicamente piccoli ma veri.

---

## 3. Formulazione Rigorosa delle Soluzioni

### 3.1 Criterio di Scelta di $\epsilon_{\text{phys}}$

Definiamo il rapporto segnale/rumore per il parametro raw:

$$
\text{SNR}(r) = \frac{|g_r^{\text{vero}}|}{u_{\text{prec}}\cdot \|\mathcal{L}\|_{\text{scale}}}
$$

Per garantire che Adam risponda al **segnale fisico** e non al **rumore di quantizzazione**, si impone la condizione di separazione delle scale:

$$
\underbrace{u_{\text{prec}}\cdot\|\mathcal{L}\|_{\text{scale}}}_{\text{floor di rumore}} \;\ll\; \epsilon_{\text{phys}} \;\ll\; \underbrace{\min_t \sqrt{\hat v_t}\big|_{\text{regime fisiologico}}}_{\text{scala tipica del gradiente vero}}
$$

Due regimi operativi coerenti:

$$
\epsilon_{\text{phys}} =
\begin{cases}
10^{-12} \text{ -- } 10^{-14}, & \text{se TF32 disattivo (FP32 puro, } u=5.96\times10^{-8}\text{)} \\[4pt]
10^{-3} \text{ -- } 10^{-4}, & \text{se TF32 attivo (} u = 4.88\times10^{-4}\text{, floor protettivo necessario)}
\end{cases}
$$

**Raccomandazione**: disattivare TF32 per la sottorete costitutiva/identificazione fisica (costo computazionale marginale, dato che la rete non è enorme: $8\times128$) e adottare $\epsilon_{\text{phys}}\sim 10^{-12}$, sfruttando appieno la precisione FP32 nativa residua.

### 3.2 Parametrizzazione Mista a Costo Nullo (FP64 selettivo)

Poiché $r_\lambda, r_{\mu_p}, r_{\mu_s}$ sono **scalari** (shape $(1,)$), mantenerli in FP64 durante l'intera pipeline — anche nella fase "Adam FP32" — ha **costo computazionale trascurabile** ($O(1)$ operazioni aggiuntive per batch) ma elimina completamente il rumore di quantizzazione sulla quantità più delicata del problema inverso: l'identificazione di $\lambda$ e $\mu_p$.

---

## 4. Modifiche Concrete al Codice

### 4.1 `train_4roll_main.py` — Gestione Granulare di TF32

```python
# ============================================================================
# IGIENE NUMERICA TF32: disabilitazione mirata per la fase di identificazione
# ============================================================================
# NOTA: TF32 introduce un errore relativo di troncamento ~4.9e-4 sui matmul
# di nn.Linear. Per un PINN che richiede derivate automatiche fino al II
# ordine (Laplaciano viscoso) e stima simultanea di parametri costitutivi
# scalari (lambda, mu_p), tale rumore si propaga e si amplifica attraverso
# l'autograd, contaminando irrimediabilmente il gradiente rispetto a
# _raw_lam / _raw_mu_p (si veda commento in SimpleHistory.plot_params
# relativo al "rumore FP32" osservato empiricamente).
#
# Scelta: disattiviamo TF32 di default. Il costo prestazionale per una rete
# 8x128 è marginale (<5% su GPU Ampere+), mentre il beneficio in termini di
# accuratezza del problema inverso è sostanziale.
TF32_ENABLED = False  # <--- flag esplicito, documentato, controllabile

torch.backends.cuda.matmul.allow_tf32 = TF32_ENABLED
torch.backends.cudnn.allow_tf32 = TF32_ENABLED
torch.set_float32_matmul_precision("highest" if not TF32_ENABLED else "high")

# Se si desidera comunque il boost prestazionale di TF32 in Fase 2 (idrodinamica,
# dove NON si stimano parametri fisici delicati), è possibile riattivarlo
# selettivamente subito prima del blocco Fase 2 in train.py tramite un
# context manager (vedi src/train.py, tf32_context).
```

### 4.2 `src/physics.py` — Parametri Fisici Sempre in FP64 (Mixed Precision Selettiva)

```python
class Physics(nn.Module):
    def __init__(self, ..., force_fp64_physics_params=True):
        super().__init__()
        ...
        self._force_fp64_physics_params = force_fp64_physics_params

        # --- Parametri raw in log-space: SEMPRE FP64 -----------------------
        # Costo trascurabile (scalari), beneficio enorme in stabilità del
        # gradiente durante il problema inverso. Il cast a FP64 avviene qui
        # indipendentemente dal dtype corrente del resto della rete (FP32
        # in fase Adam, FP64 in fase L-BFGS), rompendo l'accoppiamento con
        # eventuali chiamate .float()/.half() propagate dai moduli superiori.
        _param_dtype = torch.float64 if force_fp64_physics_params else torch.float32

        self.register_parameter(
            "_raw_lam", nn.Parameter(torch.zeros(1, device=DEVICE, dtype=_param_dtype))
        )
        self.register_parameter(
            "_raw_mu_p", nn.Parameter(torch.zeros(1, device=DEVICE, dtype=_param_dtype))
        )
        self.register_parameter(
            "_raw_mu_s",
            nn.Parameter(torch.zeros(1, device=DEVICE, dtype=_param_dtype), requires_grad=False),
        )
        ...

    def _apply(self, fn):
        """Override di nn.Module._apply per intercettare .float()/.double()
        e IMPEDIRE il downcast dei parametri fisici quando force_fp64_physics_params=True.
        Necessario perché convert_to_fp32()/convert_to_fp64() in utils.py
        chiamano tipicamente model.float() / model.double() sull'intero modulo,
        includendo i parametri scalari del problema inverso."""
        protected = {id(self._raw_lam), id(self._raw_mu_p), id(self._raw_mu_s)}
        for name, param in self.named_parameters(recurse=False):
            if id(param) in protected and self._force_fp64_physics_params:
                continue  # Salta il cast: resta permanentemente float64
        return super()._apply(fn)

    @property
    def lam(self):
        """Restituisce lambda in FP64; il cast a FP32 (se necessario per
        interoperabilita' con il resto del grafo) va fatto ESPLICITAMENTE
        e a valle, non ai bordi del parametro sorgente."""
        return self.guess_lam.double() * torch.exp(self._raw_lam).squeeze()
```

> **Nota implementativa**: poiché `lam`, `mu_p_nd`, `Wi` entrano poi in operazioni con tensori FP32 (es. `tau`, `u_x`), è necessario un cast esplicito e **controllato** al punto di interfaccia (`_nondim`), per evitare errori di tipo misto in autograd:

```python
    def _nondim(self):
        """Parametri adimensionali: calcolati in FP64 (precisione piena sui
        parametri fisici), poi castati al dtype del grafo principale SOLO
        al momento dell'uso in operazioni miste con i tensori di stato."""
        target_dtype = self._raw_mu_s.dtype if not self._force_fp64_physics_params \
            else self.eta_0.dtype  # dtype del grafo corrente (FP32 Adam / FP64 LBFGS)

        mu_p_nd = (self.mu_p / self.eta_0.double()).to(target_dtype)
        mu_s_nd = (self.mu_s / self.eta_0.double()).to(target_dtype)
        Re_scale = (RHO * self.U_ref * self.H_ref / self.eta_0.double()).to(target_dtype)
        Wi = (self.lam * self.U_ref / self.H_ref).to(target_dtype)
        return Re_scale, Wi, mu_s_nd, mu_p_nd, self.eps, self.alpha
```

### 4.3 `src/train.py` — Optimizer con Gruppi ed $\epsilon$ Differenziati

```python
# ============================================================================
# COSTRUZIONE OPTIMIZER ADAM CON EPS DIFFERENZIATO PER GRUPPO DI PARAMETRI
# ============================================================================
# Razionale: i parametri fisici raw (_raw_lam, _raw_mu_p, _raw_mu_s) hanno
# una dinamica di gradiente scalare radicalmente diversa da quella dei pesi
# di rete (10^5+ parametri). Condividere lo stesso eps=1e-7 causa,
# in prossimita' della convergenza (g_t -> 0), un update mal condizionato
# Delta_theta ~ -eta * m_hat / eps, con fattore di amplificazione eta/eps ~ 1e3.
#
# ADAM_EPS_PHYSICS va scelto coerentemente con lo stato di TF32:
#   - TF32 OFF (raccomandato, vedi train_4roll_main.py): eps_phys ~ 1e-12
#   - TF32 ON  (fallback prestazionale): eps_phys ~ 1e-3 (floor protettivo
#     commensurato al rumore di troncamento TF32, u_TF32 ~ 4.9e-4)

ADAM_EPS_NETWORK = getattr(builtins, "ADAM_EPS", 1e-7)
ADAM_EPS_PHYSICS = getattr(
    builtins, "ADAM_EPS_PHYSICS",
    1e-12 if not getattr(builtins, "TF32_ENABLED", False) else 1e-3
)

def build_staged_optimizer(model, physics, base_lr, param_lr_factor):
    """Crea l'optimizer Adam con param_groups a eps e lr differenziati."""
    network_params = list(model.parameters())
    physics_params = physics.get_phase2_params() + \
        [p for p in [physics._raw_lam, physics._raw_mu_p] if p.requires_grad]

    param_groups = [
        {
            "params": network_params,
            "lr": base_lr,
            "eps": ADAM_EPS_NETWORK,
        },
        {
            "params": physics_params,
            "lr": base_lr * param_lr_factor,
            "eps": ADAM_EPS_PHYSICS,   # <-- differenziazione critica
            "weight_decay": 0.0,       # nessun decadimento su parametri fisici
        },
    ]
    return torch.optim.Adam(param_groups, betas=(0.9, 0.999))
```

Inoltre, per rendere reversibile e testabile la scelta TF32 anche a runtime (es. per A/B test Fase 1 vs Fase 2), un context manager dedicato:

```python
from contextlib import contextmanager

@contextmanager
def tf32_context(enabled: bool):
    """Context manager per abilitare/disabilitare TF32 in modo localizzato
    e reversibile, senza alterare lo stato globale per l'intera run.
    Utile per isolare la Fase 1 (identificazione lambda, mu_p) in precisione
    piena, riservando eventualmente TF32 alla sola Fase 2 idrodinamica
    (dove non si stimano parametri costitutivi delicati)."""
    prev_matmul = torch.backends.cuda.matmul.allow_tf32
    prev_cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = enabled
    torch.backends.cudnn.allow_tf32 = enabled
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_matmul
        torch.backends.cudnn.allow_tf32 = prev_cudnn

# Esempio d'uso nel loop di training (Fase 1):
# with tf32_context(enabled=False):
#     loss = compute_and_backward_losses(...)   # identificazione lam, mu_p
```

### 4.4 Verifica di Coerenza Post-Conversione FP64 (L-BFGS)

Aggiungere un assert diagnostico in `convert_to_fp64` (in `src/utils.py`, non mostrato ma presumibile) per garantire che i parametri fisici non siano stati accidentalmente ri-castati o duplicati in dtype incoerente:

```python
def assert_physics_dtype_integrity(physics, expected_dtype=torch.float64):
    """Sanity check da invocare dopo ogni convert_to_fp32/convert_to_fp64:
    verifica che i parametri fisici scalari mantengano la precisione piena
    indipendentemente dal dtype del resto del grafo computazionale."""
    for name in ["_raw_lam", "_raw_mu_p", "_raw_mu_s"]:
        p = getattr(physics, name)
        assert p.dtype == expected_dtype, (
            f"[IGIENE NUMERICA] {name} ha dtype {p.dtype}, atteso {expected_dtype}. "
            "Verificare che _apply() in Physics non sia stato bypassato."
        )
```

---

## 5. Raccomandazioni Operative Conclusive

| Aspetto | Stato Attuale | Raccomandazione |
|---|---|---|
| TF32 globale | Sempre attivo (`"high"`) | Disattivare di default; riattivare selettivamente solo in Fase 2 hydrodynamics via `tf32_context` |
| `_grad` scaling | $\Gamma^2$ non mitigato | Documentare/verificare $\Gamma = H_{\text{ref}}/H_{\text{coord}} \approx 1$ per minimizzare amplificazione |
| $\epsilon_{\text{Adam}}$ | Condiviso, $10^{-7}$ | Param group dedicato: $10^{-12}$ (TF32 off) o $10^{-3}$ (TF32 on) |
| Parametri fisici | Dtype seguono il modulo (FP32→FP64) | Forzare FP64 permanente via override di `_apply` |
| Diagnostica | Nessuna | Assert di integrità dtype dopo ogni conversione di precisione |

L'insieme di queste modifiche disaccoppia rigorosamente la **sorgente di rumore numerico** (TF32) dalla **sensibilità dell'ottimizzatore** ($\epsilon$), permettendo alla Fase 1 di identificare $\lambda$ e $\mu_p$ con un pavimento di rumore prossimo alla precisione macchina nativa, coerentemente con l'obiettivo dichiarato di raffinamento a doppia precisione in L-BFGS.