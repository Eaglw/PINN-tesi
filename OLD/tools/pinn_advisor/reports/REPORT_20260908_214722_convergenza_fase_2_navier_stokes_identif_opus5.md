# Report Analisi PINN: Convergenza Fase 2 Navier Stokes Identificabilita Mu_S E Pressione Problema Inverso

- **Data/Ora:** `2026-09-08 22:45:31`
- **Modello:** `claude-opus-5-default`
- **Topic:** `convergenza fase 2 navier stokes identificabilita mu_s e pressione problema inverso`
- **Target Codebase:** `final_roll/src/`

---

# Analisi rigorosa della Fase 2: identificabilità di $\mu_s$, gauge di pressione e strategie VarPro

---

## 0. Sintesi esecutiva (diagnosi in cinque righe)

Il problema che state osservando **non è un problema di ottimizzazione: è una degenerazione strutturale del funzionale**. Nella Fase 2, con $\boldsymbol\tau$ congelato, il residuo del momento è **affine in $\mu_s$ e affine in $p$**, e la rete di pressione — essendo un approssimatore universale vincolato solo in un punto — genera un sottospazio $\mathcal G=\{\nabla q\}$ che **assorbe quasi interamente** la colonna di sensitività $\partial_{\mu_s}\mathbf R=-\Delta\mathbf u$. L'informazione residua su $\mu_s$ è **esattamente** la norma della proiezione di Hodge–Leray di $\Delta\mathbf u$ sul complemento solenoidale, cioè è controllata da $\Delta\omega$ (bilaplaciano di $\psi$). Le cinque azioni raccomandate:

1. **Eliminare $p$ analiticamente** (VarPro / proiezione di Leray) prima di stimare $\mu_s$ — equivalente alla formulazione in vorticità, ma implementabile **senza derivate quarte** tramite VarPro sull'ultimo layer lineare di `model_p`.
2. **Ancoraggio hard** della pressione (non in loss): $p(\mathbf x)=p_{\text{scale}}\big(\hat p(\mathbf x)-\hat p(\mathbf x_0)\big)+p_{\text{ref}}$.
3. **Trust-region in spazio funzionale su $\psi$** (proximal / EWC-Fisher), non semplice "mobilità libera".
4. **Riparametrizzare in $\mu_{tot}$**: con $\boldsymbol\tau$ congelato l'errore di Fase 1 su $\mu_p$ si trasferisce con bias $-1{:}1$ su $\hat\mu_s$; solo $\mu_{tot}$ è robustamente identificabile.
5. **Calcolare l'indice di identificabilità $\rho_{id}$ e il CRLB *prima* di lanciare la Fase 2**: se $\rho_{id}<10^{-2}$ nessun ottimizzatore vi salverà.

---

## 1. Struttura algebrica del problema di Fase 2

### 1.1 Riformulazione affine

Sia $\Omega\subset\mathbb R^2$, $\mathbf u=\nabla^\perp\psi:=(\partial_y\psi,-\partial_x\psi)$, $\omega=\partial_xv-\partial_yu=-\Delta\psi$. Il residuo del momento è

$$
\mathbf R(\mathbf x;\psi,p,\mu_s)\;=\;\underbrace{\rho(\mathbf u\cdot\nabla)\mathbf u-\nabla\cdot\boldsymbol\tau^{\text{froz}}}_{=:\;\mathbf b(\mathbf x;\psi)}\;+\;\nabla p\;-\;\mu_s\,\underbrace{\Delta\mathbf u}_{=:\;\mathbf a(\mathbf x;\psi)} .
$$

**Osservazione chiave.** A $\psi$ fissato, $\mathbf R$ è **affine** nella coppia $(\,p,\mu_s\,)$. Il funzionale di Fase 2 (parte momento) è quindi una **forma quadratica convessa** in $(p,\mu_s)$:

$$
\mathcal L_{\text{mom}}(p,\mu_s)=\tfrac12\big\|\mathbf b+\nabla p-\mu_s\mathbf a\big\|^2_{L^2(\Omega)} .
$$

Tutta la patologia sta nell'**angolo tra la colonna $\mathbf a$ e il sottospazio $\mathcal G$**.

### 1.2 Decomposizione di Helmholtz–Hodge e teorema di non-identificabilità

Sia la decomposizione ortogonale in $L^2(\Omega)^2$:

$$
L^2(\Omega)^2=\mathcal H\oplus\mathcal G,\qquad
\mathcal H=\{\mathbf w:\nabla\!\cdot\!\mathbf w=0,\ \mathbf w\!\cdot\!\mathbf n|_{\partial\Omega}=0\},\qquad
\mathcal G=\{\nabla q: q\in H^1(\Omega)\},
$$

con $\mathbb P_{\mathcal H}$ il proiettore di Leray. Minimizzando **esattamente** rispetto a $p$ (VarPro):

$$
\boxed{\;\nabla p^\star(\mu_s)=-\mathbb P_{\mathcal G}\big[\mathbf b-\mu_s\mathbf a\big],\qquad
\mathcal L^\star_{\text{mom}}(\mu_s)=\tfrac12\big\|\mathbb P_{\mathcal H}\mathbf b-\mu_s\,\mathbb P_{\mathcal H}\mathbf a\big\|^2 \;}
$$

> **Teorema 1 (non-identificabilità).** A $\psi,\boldsymbol\tau$ fissati e con $p$ libero (a meno di una costante), $\mu_s$ è identificabile dal residuo del momento **se e solo se** $\mathbb P_{\mathcal H}\Delta\mathbf u\neq 0$. Poiché $\Delta\mathbf u=-\nabla^\perp\omega$ è già a divergenza nulla, si ha $\mathbb P_{\mathcal H}\Delta\mathbf u=0$ **se e solo se** $\Delta\mathbf u=\nabla h$ con $\Delta h=0$, ossia **se e solo se $\Delta\omega\equiv 0$** in $\Omega$ (e la traccia normale è compatibile con un armonico).
>
> *Dimostrazione.* $\operatorname{curl}(\Delta\mathbf u)=\Delta(\operatorname{curl}\mathbf u)=\Delta\omega$. Un campo $L^2$ è in $\mathcal G$ sse il suo rotore distribuzionale è nullo e la sua parte armonica coincide col campo. Dato $\nabla\!\cdot\!\Delta\mathbf u=0$, l'unica componente di $\mathcal G$ ammissibile è un gradiente armonico. $\square$

**Conseguenza operativa immediata.** La stima di $\mu_s$ **non usa** la parte irrotazionale del bilancio di momento: quella è *interamente* consumata dalla pressione. Tutta l'informazione vive nell'**equazione di trasporto della vorticità**:

$$
\boxed{\;\rho\,(\mathbf u\cdot\nabla)\omega \;=\; \mu_s\,\Delta\omega \;+\; \mathcal T[\boldsymbol\tau],\qquad
\mathcal T=\partial_{xx}\tau_{xy}-\partial_{yy}\tau_{xy}+\partial_{xy}(\tau_{yy}-\tau_{xx}) \;}
$$

### 1.3 Matrice d'informazione di Fisher e CRLB

Modello statistico dei residui di collocazione: $\mathbf R(\mathbf x_i)=\boldsymbol\varepsilon_i\sim\mathcal N(0,\sigma^2 I_2)$ i.i.d. (interpretazione bayesiana della PDE-loss). Sia $\hat p_\theta(\mathbf x)=\sum_{j=1}^M c_j\phi_j(\mathbf x)+c_0$ (ultimo layer lineare di `model_p`), $\Phi\in\mathbb R^{2N\times M}$ la matrice dei $\nabla\phi_j$ impilata sulle componenti, $\mathbf a\in\mathbb R^{2N}$ la colonna $-\Delta\mathbf u$. Con $\theta=(c,\mu_s)$:

$$
\mathcal I(\theta)=\frac{1}{\sigma^2}
\begin{bmatrix}
\Phi^\top\Phi & -\Phi^\top \mathbf a\\[2pt]
-\mathbf a^\top\Phi & \ \ \mathbf a^\top\mathbf a
\end{bmatrix}
$$

**Informazione efficace su $\mu_s$** = complemento di Schur:

$$
\boxed{\;
\mathcal I_{\text{eff}}(\mu_s)=\frac{1}{\sigma^2}\Big[\|\mathbf a\|^2-\mathbf a^\top\Phi(\Phi^\top\Phi)^{-1}\Phi^\top\mathbf a\Big]
=\frac{\big\|\mathbb P^{\perp}_{\mathcal G_M}\Delta\mathbf u\big\|^2}{\sigma^2}
\;\xrightarrow[M\to\infty]{}\;\frac{\big\|\mathbb P_{\mathcal H}\Delta\mathbf u\big\|^2}{\sigma^2}\;}
$$

e il **Cramér–Rao**:

$$
\operatorname{Var}(\hat\mu_s)\ \ge\ \frac{\sigma^2}{\|\mathbb P_{\mathcal H}\Delta\mathbf u\|^2_{L^2(\Omega)}} ,
\qquad
\frac{\operatorname{sd}(\hat\mu_s)}{\mu_s}\ \ge\ \frac{\sigma}{\mu_s\,\|\mathbb P_{\mathcal H}\Delta\mathbf u\|}.
$$

Definisco l'**indice di identificabilità** (coseno del complemento angolare, $1/\sqrt{\text{VIF}}$):

$$
\boxed{\;\rho_{id}:=\frac{\|\mathbb P^{\perp}_{\mathcal G_M}\Delta\mathbf u\|}{\|\Delta\mathbf u\|}\in[0,1]\;}
$$

Con $\Delta\mathbf u=-\nabla^\perp\omega$ e la disuguaglianza di Poincaré su $\Omega$:

$$
\|\mathbb P_{\mathcal H}\Delta\mathbf u\|_{L^2}\;\simeq\;\|\Delta\omega\|_{H^{-1}(\Omega)}\;\ge\;C_\Omega^{-1}\,\|\Delta\omega\|_{L^2}\cdot\ \text{(fattore di scala)} .
$$

### 1.4 Stima fisica dell'ordine di grandezza (perché il vostro problema è duro)

Nel limite di fluido di secondo ordine ($Wi\ll1$), $\boldsymbol\tau\simeq 2\mu_p\mathbf D-2\mu_p\lambda\,\overset{\triangledown}{\mathbf D}$, da cui $\mathcal T\simeq\mu_p\Delta\omega+O(\mu_p\lambda)$ e l'equazione di vorticità *esatta* dà

$$
\Delta\omega=\frac{\rho(\mathbf u\cdot\nabla)\omega-2\mu_p\lambda\,\operatorname{curl}(\nabla\!\cdot\!\overset{\triangledown}{\mathbf D})}{\mu_s+\mu_p}
\quad\Longrightarrow\quad
\|\Delta\omega\|\sim \frac{U}{H^3}\,\mathcal O\!\big(Re+Wi\big).
$$

> **Corollario 1.** L'informazione su $\mu_s$ scala come $\mathcal I_{\text{eff}}\propto (Re+Wi)^2$. **Nel limite Stokes–Newtoniano ($Re\to0$, $Wi\to0$) $\psi$ è biarmonica, $\Delta\omega\equiv0$ e $\mu_s$ è *esattamente* non identificabile** (degenerazione classica della scala di viscosità nello Stokes omogeneo con soli dati Dirichlet di velocità: $(\mathbf u,p,\mu)\mapsto(\mathbf u,\kappa p,\kappa\mu)$ è una simmetria esatta).
>
> Nel four-roll mill viscoelastico l'identificabilità è dunque **portata dalla non-Newtonianità dello stress e dall'inerzia**, non dalla cinematica di per sé.

Questo spiega perfettamente la fenomenologia riportata: Fase 1 (dove $\lambda,\mu_p$ entrano nell'equazione costitutiva, con sensitività $O(1)$) converge; Fase 2 (dove $\mu_s$ entra solo tramite $\Delta\omega=O(Re+Wi)$, per giunta schermato dalla pressione) no.

---

## 2. La degenerazione supplementare: `model_psi` mobile

### 2.1 Gauge $(\mu_s,\psi,p)$

Se in Fase 2 $\psi$ è libero e i pesi dati/BC sono deboli, esiste una **famiglia a un parametro di minimi esatti**. Infatti per ogni $\mu_s>0$ si può risolvere

$$
\rho(\mathbf u\cdot\nabla)\omega-\mu_s\Delta\omega=\mathcal T[\boldsymbol\tau^{\text{froz}}]\quad\text{in }\Omega,
$$

ottenendo $\psi_{\mu_s}$ (problema di Navier–Stokes con forzante fissata: **ben posto** per BC assegnate) e poi $p_{\mu_s}$ per quadratura. Dunque

$$
\mathcal L_{\text{mom}}\big(\psi_{\mu_s},p_{\mu_s},\mu_s\big)=0\qquad\forall\,\mu_s>0 .
$$

**La loss di momento, da sola, non identifica $\mu_s$: identifica solo la varietà $\{(\mu_s,\psi_{\mu_s})\}$.** L'unica cosa che rompe la degenerazione è il **vincolo cinematico** (dati di velocità e/o BC dei rulli) che pinza $\psi$.

### 2.2 Sensitività cinematica e FIM totale

Differenziando la relazione precedente rispetto a $\mu_s$ (operatore linearizzato $\mathcal A$ di Oldroyd–B linearizzato con $\boldsymbol\tau$ congelato ⇒ Navier–Stokes linearizzato):

$$
\mathcal A\,\delta\psi=\Delta\omega\;\delta\mu_s,\qquad
\mathcal A\,\varphi:=\rho\big[(\mathbf u\cdot\nabla)(-\Delta\varphi)+(\nabla^\perp\varphi\cdot\nabla)\omega\big]-\mu_s\Delta^2\varphi,
$$

quindi la **sensitività osservabile in velocità** è

$$
\mathbf s(\mathbf x):=\frac{\partial\mathbf u}{\partial\mu_s}=\nabla^\perp\mathcal A^{-1}\Delta\omega .
$$

FIM totale con pesi $W_{data},W_{mom}$ (e $p$ profilata):

$$
\boxed{\;
\mathcal I_{\text{tot}}(\mu_s)=\frac{W_{data}}{\sigma_d^2}\,\|\mathbf s\|^2_{L^2(\Omega_d)}
\;+\;\frac{W_{mom}}{\sigma_r^2}\,\big\|\mathbb P_{\mathcal H}\Delta\mathbf u\big\|^2_{L^2(\Omega)}\;}
$$

**Interpretazione.** Vi sono **due canali informativi indipendenti**: (i) il *canale cinematico* (quanto la velocità osservata cambia se cambio $\mu_s$) e (ii) il *canale residuale* (quanto il bilancio di vorticità è violato). Se $\psi$ è **congelato**, il canale (i) scompare ma il canale (ii) diventa "pulito"; se $\psi$ è **libero e non regolarizzato**, il canale (ii) collassa a zero (la rete "accomoda" $\psi$) e resta solo (i), che però ha guadagno $\|\mathcal A^{-1}\|\cdot\|\Delta\omega\|$ — piccolo e mal condizionato.

> **Raccomandazione (risposta al quesito 2).** Non lasciare $\psi$ né completamente libero né completamente congelato: usare una **trust-region in spazio funzionale** attorno alla soluzione di Fase 1, con raggio calibrato sul livello di residuo cinematico di Fase 1:
> $$
> \mathcal L_{\text{prox}}=\frac{\gamma}{|\Omega|}\int_\Omega\big|\nabla^\perp(\psi-\psi^{(1)})\big|^2\,d\mathbf x
> =\gamma\,\big\|\mathbf u_\theta-\mathbf u^{(1)}\big\|^2_{L^2(\Omega)} ,
> $$
> preferibile all'EWC nello spazio dei pesi perché **invariante per riparametrizzazione della rete**. Formulazione rigorosa: **problema vincolato**
> $$
> \min_{\psi,p,\mu_s}\ \mathcal L^\star_{\text{mom}}\quad\text{s.t.}\quad \mathcal L_{data}\le\epsilon_d,\ \ \mathcal L_{bc}\le\epsilon_b,\ \ \|\mathbf u-\mathbf u^{(1)}\|^2\le\epsilon_\psi ,
> $$
> risolto con **Lagrangiana aumentata** (moltiplicatori aggiornati con ascesa duale), che sostituisce l'arbitrarietà dei pesi $W$ con soglie $\epsilon$ **fisicamente interpretabili** (es. $\epsilon_d = $ varianza del rumore dei dati COMSOL interpolati).

### 2.3 Bias da $\boldsymbol\tau$ congelato: solo $\mu_{tot}$ è robusto

Sia $\boldsymbol\tau^{\text{froz}}=\boldsymbol\tau^{true}+\delta\boldsymbol\tau$. Lo stimatore VarPro-in-$p$ (§3) è lineare:

$$
\hat\mu_s=\frac{\langle \Delta\omega,\ \rho(\mathbf u\cdot\nabla)\omega-\mathcal T[\boldsymbol\tau^{\text{froz}}]\rangle}{\|\Delta\omega\|^2}
=\mu_s^{true}-\frac{\langle\Delta\omega,\ \mathcal T[\delta\boldsymbol\tau]\rangle}{\|\Delta\omega\|^2}.
$$

Se l'errore di Fase 1 su $\mu_p$ produce $\delta\boldsymbol\tau\approx 2\,\delta\mu_p\,\mathbf D$ (che è la componente **dominante** dell'errore, essendo il modo Newtoniano quello più eccitato), allora $\mathcal T[\delta\boldsymbol\tau]=\delta\mu_p\Delta\omega$ e

$$
\boxed{\;\hat\mu_s=\mu_s^{true}-\delta\mu_p\quad\Longrightarrow\quad \hat\mu_s+\hat\mu_p^{(1)}=\mu_{tot}^{true}\;}
$$

**Trasferimento di bias $1{:}1$.** Questa è, con ogni probabilità, la ragione per cui vedete $\hat\mu_s$ "andare a sbattere" contro valori negativi o assurdi quando $\hat\mu_p^{(1)}$ è sovrastimato di poco. Corollari operativi:

* **Riparametrizzare**: `_raw_mu_tot` addestrabile in Fase 2, con $\mu_s=\mu_{tot}-\mu_p^{(1)}$ e vincolo hard $\mu_s>0$ via softplus sul residuo. Il numero di condizione del problema è $O(1)$ in $\mu_{tot}$, $O(\mu_{tot}/|\delta\mu_p|)$ in $\mu_s$.
* **Fase 3 congiunta**: sbloccare $(\mu_p,\lambda,\mu_s)$ e $\boldsymbol\tau$ con *tutte* le equazioni attive, partendo dal punto di Fase 2. Solo il problema congiunto (costitutiva + momento) rompe la collinearità $\mu_s\!\leftrightarrow\!\mu_p$, e lo fa attraverso i termini $O(Wi)$ di $\boldsymbol\tau$ (§1.4).

---

## 3. Confronto rigoroso delle tre formulazioni (quesito 3)

| Formulazione | Incognite | Ordine derivate su $\psi$ | Degenerazione $p\!-\!\mu_s$ | Norma implicita del residuo | Verdetto |
|---|---|---|---|---|---|
| **(A)** Momento in $\nabla p$, $p$ rete libera | $\psi,p,\mu_s$ | 3 | **presente** (Teor. 1) — $\mathcal I_{\text{eff}}$ ridotta dal fattore $\rho_{id}^2$ | $L^2$ | ❌ (causa del vostro stallo) |
| **(B)** Curl del momento (vorticità) | $\psi,\mu_s$ | **4** ($\Delta^2\psi$) | assente per costruzione | $\dot H^{1}$ del momento ⇒ **sovrappesa le alte frequenze** | ⚠️ corretta ma numericamente fragile in autograd |
| **(C)** **VarPro su $p$** (proiezione di Leray discreta) | $\mu_s$ (+ $c$ chiuso) | **2** | assente | $L^2$ **corretta** | ✅ **Raccomandata** |

> **Teorema 2 (equivalenza VarPro ≡ Leray ≡ curl).** Sia $\mathcal G_M=\text{span}\{\nabla\phi_j\}_{j=1}^M$. Minimizzare $\|\mathbf b+\nabla p-\mu_s\mathbf a\|^2_{L^2}$ congiuntamente in $(c,\mu_s)$ è equivalente a minimizzare $\|\mathbb P^\perp_{\mathcal G_M}(\mathbf b-\mu_s\mathbf a)\|^2$. Per $M\to\infty$ con $\mathcal G_M$ denso in $\mathcal G$, $\mathbb P^\perp_{\mathcal G_M}\to\mathbb P_{\mathcal H}$, e poiché un campo di $\mathcal H$ è univocamente determinato dal suo rotore (con $\mathbf w\cdot\mathbf n=0$), la funzione obiettivo è equivalente a $\|\Delta^{-1}\!\operatorname{curl}(\mathbf b-\mu_s\mathbf a)\|$, cioè al residuo di vorticità **misurato in $H^{-1}$** — che è la norma *corretta*, a differenza della (B) che lo misura in $L^2$.

**Questo è il punto tecnico centrale della mia raccomandazione:** la formulazione VarPro (C) vi dà i benefici della formulazione in vorticità (eliminazione esatta di $p$) **senza calcolare derivate quarte** e **senza distorcere lo spettro del residuo**, perché la proiezione è realizzata nello spazio finito-dimensionale delle feature dell'ultimo layer di `model_p`. Inoltre il passo VarPro è il **passo di Gauss–Newton esatto** sul blocco lineare: convergenza quadratica su quel blocco, cosa che Adam non raggiungerà mai.

---

## 4. Algoritmo proposto: Fase 2 = 2a + 2b + 2c

**Fase 2a — Identificazione (pressione eliminata).**
Alternanza:
* passo VarPro (ogni $K$ iterazioni, closed-form): risolvi il LS lineare in $(c,\mu_s)$ ⇒ aggiorna ultimo layer di `model_p` e `_raw_mu_s`;
* passi Adam su $(\theta_\psi,\theta_p^{\text{hidden}})$ con Lagrangiana aumentata su dati/BC/proximal.

**Fase 2b — Ricostruzione della pressione (problema convesso).**
Con $\mu_s$ e $\psi$ congelati, $\mathbf g:=-\rho(\mathbf u\cdot\nabla)\mathbf u+\mu_s\Delta\mathbf u+\nabla\!\cdot\!\boldsymbol\tau$ è noto; minimizza $\int_\Omega|\nabla p-\mathbf g|^2$ con ancoraggio hard. È il problema di Neumann $\Delta p=\nabla\!\cdot\!\mathbf g$, $\partial_n p=\mathbf g\cdot\mathbf n$: **strettamente convesso** modulo la costante (fissata analiticamente). Nessuna interazione con $\mu_s$.

**Fase 2c — Raffinamento congiunto** con L-BFGS in `float64`, tutti i parametri liberi, moltiplicatori congelati.

---

## 5. Codice pronto per `final_roll/src/`

### 5.1 `src/models.py` (o `CombinedModel` in `train.py`): ancoraggio hard della pressione

```python
class CombinedModel(nn.Module):
    def __init__(self, p_scale=1.0, tau_scale=1.0, x_anchor=None, p_ref=0.0):
        ...
        self.register_buffer("x_anchor", torch.as_tensor(x_anchor, dtype=torch.float32,
                                                         device=DEVICE).reshape(1, -1))
        self.register_buffer("p_ref", torch.tensor(float(p_ref), device=DEVICE))
        self.hard_anchor = x_anchor is not None

    def pressure(self, x):
        """p con gauge fissata in modo HARD: nessun termine di loss, nessuna direzione nulla."""
        p_raw = self.model_p(x)
        if self.hard_anchor:
            p0 = self.model_p(self.x_anchor)          # (1,1)
            return self.p_scale * (p_raw - p0) + self.p_ref
        return self.p_scale * p_raw
```

> Motivazione: il vincolo puntuale $p(\mathbf x_0)=p_{ref}$ ha **misura nulla**; nella loss produce una direzione con autovalore $\sim 1/N$ nell'Hessiana, che Adam ignora ⇒ la costante di pressione va a deriva e "inquina" i gradienti tramite il coupling con i pesi condivisi. L'ancoraggio hard **rimuove esattamente** il modo nullo.

### 5.2 `src/physics.py`: cinematica di secondo ordine + parti affini del momento

```python
    def kinematics2(self, model, x):
        """u, v, gradienti primi e Laplaciani da psi (solo derivate <= 2 su u)."""
        if not x.requires_grad:
            x = x.clone().requires_grad_(True)
        psi = model.model_psi(x) * (self.H_coord / self.H_ref)
        g   = self._grad(psi, x)
        u, v = g[:, 1:2], -g[:, 0:1]
        du, dv = self._grad(u, x), self._grad(v, x)
        ux, uy, vx, vy = du[:, 0:1], du[:, 1:2], dv[:, 0:1], dv[:, 1:2]
        uxx = self._grad(ux, x)[:, 0:1]; uyy = self._grad(uy, x)[:, 1:2]
        vxx = self._grad(vx, x)[:, 0:1]; vyy = self._grad(vy, x)[:, 1:2]
        return dict(x=x, u=u, v=v, ux=ux, uy=uy, vx=vx, vy=vy,
                    lap_u=uxx + uyy, lap_v=vxx + vyy,
                    omega=vx - uy)

    def momentum_affine_parts(self, model, x):
        r"""Restituisce (b, a, K) con  R = b + grad p - mu_s * a,
            b = rho (u.grad)u - div(tau_froz),   a = Lap u."""
        K = self.kinematics2(model, x); x = K['x']
        tau = model.model_tau(x) * model.tau_scale
        txx, txy, tyy = tau[:, 0:1], tau[:, 1:2], tau[:, 2:3]
        dtxx, dtxy, dtyy = self._grad(txx, x), self._grad(txy, x), self._grad(tyy, x)
        divt_x = dtxx[:, 0:1] + dtxy[:, 1:2]
        divt_y = dtxy[:, 0:1] + dtyy[:, 1:2]
        b_x = RHO * (K['u'] * K['ux'] + K['v'] * K['uy']) - divt_x
        b_y = RHO * (K['u'] * K['vx'] + K['v'] * K['vy']) - divt_y
        return torch.cat([b_x, b_y], 0), torch.cat([K['lap_u'], K['lap_v']], 0), K
```

### 5.3 `src/physics.py`: **VarPro sull'ultimo layer** (cuore della soluzione)

```python
from torch.func import jacrev, vmap

    def _p_features_and_grads(self, model, x):
        """phi_j(x) e grad phi_j(x) per l'ultimo layer lineare di model_p."""
        trunk = model.model_p.network[:-1]          # tutto tranne l'ultimo Linear
        f = lambda z: trunk(z.unsqueeze(0)).squeeze(0)          # R^2 -> R^M
        F  = trunk(x)                                            # (N, M)
        dF = vmap(jacrev(f))(x)                                  # (N, M, 2)
        dF = dF * (self.H_ref / self.H_coord)                    # coerenza con self._grad
        return F, dF

    def varpro_pressure_and_mus(self, model, x_coll, lam_tik=1e-8,
                                mu_s_prior=None, w_prior=0.0, update=True):
        r"""Passo di Gauss-Newton ESATTO sul blocco lineare (c, mu_s).

        Risolve   min_{c, mu_s} || Phi c - mu_s a + b ||^2 + lam ||c||^2 + w (mu_s-mu_s^0)^2
        dove Phi_{ij} = p_scale * d_x phi_j,  a = Lap u,  b = rho(u.grad)u - div tau.
        Ritorna dict con mu_s stimato e diagnostica di identificabilita'.
        """
        with torch.enable_grad():
            b, a, K = self.momentum_affine_parts(model, x_coll)
            F, dF   = self._p_features_and_grads(model, K['x'])

        b = b.detach(); a = a.detach(); dF = dF.detach(); F = F.detach()
        N, M = F.shape
        s = model.p_scale
        Phi = torch.cat([dF[:, :, 0], dF[:, :, 1]], 0) * s        # (2N, M)

        A = torch.cat([Phi, -a], dim=1)                            # (2N, M+1)
        rhs = -b                                                   # (2N, 1)

        # normal equations regolarizzate (Tikhonov solo su c, prior debole su mu_s)
        AtA = A.T @ A
        Atb = A.T @ rhs
        reg = torch.eye(M + 1, device=A.device, dtype=A.dtype) * lam_tik
        reg[M, M] = w_prior
        if mu_s_prior is not None:
            Atb[M] = Atb[M] + w_prior * mu_s_prior
        z = torch.linalg.solve(AtA + reg, Atb)                     # (M+1,1)
        c, mu_s_hat = z[:M, 0], z[M, 0]

        # ---------- diagnostica di identificabilita' (Schur / CRLB) ----------
        G  = Phi.T @ Phi + lam_tik * torch.eye(M, device=A.device, dtype=A.dtype)
        Pa = Phi @ torch.linalg.solve(G, Phi.T @ a)                # proiezione di a su span(grad phi)
        a_perp = a - Pa
        schur  = (a_perp ** 2).sum()
        rho_id = torch.sqrt(schur / ((a ** 2).sum() + 1e-30))
        resid  = A @ z - rhs
        sigma2 = (resid ** 2).sum() / max(2 * N - M - 1, 1)
        sd_mus = torch.sqrt(sigma2 / (schur + 1e-30))              # CRLB

        if update and torch.isfinite(mu_s_hat):
            mu_s_pos = torch.clamp(mu_s_hat, min=1e-6)
            with torch.no_grad():
                model.model_p.network[-1].weight.copy_(c.view(1, -1))
                # bias: ancoraggio hard e' gestito in model.pressure(); qui bias = 0
                model.model_p.network[-1].bias.zero_()
                self._raw_mu_s.copy_(torch.log(mu_s_pos / self.guess_mu_s).reshape(1))

        return dict(mu_s=mu_s_hat.item(), rho_id=rho_id.item(),
                    sd_mu_s=sd_mus.item(), rel_sd=(sd_mus / (mu_s_hat.abs() + 1e-12)).item(),
                    resid=torch.sqrt((resid ** 2).mean()).item())
```

**Uso diagnostico obbligatorio prima della Fase 2:**

```python
diag = physics.varpro_pressure_and_mus(model, x_coll_big, update=False)
print(f"rho_id = {diag['rho_id']:.3e}   mu_s_LS = {diag['mu_s']:.4f} "
      f"+/- {diag['sd_mu_s']:.4f}  (rel {diag['rel_sd']:.1%})")
```

**Criteri di accettazione:**

| $\rho_{id}$ | Interpretazione | Azione |
|---|---|---|
| $>0.2$ | ben identificabile | procedere normalmente |
| $0.02\!-\!0.2$ | marginale | VarPro obbligatorio + fp64 + prior su $\mu_{tot}$ |
| $<0.02$ | **non identificabile** | riparametrizzare in $\mu_{tot}$, aggiungere dati di trazione/coppia, aumentare $Wi$ o $Re$ |

### 5.4 Residuo di momento *proiettato* per il training a gradiente (Fase 2a)

Anche fra due passi VarPro conviene che la loss usata da Adam sia **già proiettata**, altrimenti i gradienti su $\theta_\psi$ contengono la componente irrotazionale spuria:

```python
    def loss_momentum_projected(self, model, x_coll, lam_tik=1e-8):
        """|| P_perp ( b - mu_s a ) ||^2 : la pressione e' eliminata analiticamente.
        I gradienti fluiscono su theta_psi, theta_tau(congelato) e _raw_mu_s."""
        b, a, K = self.momentum_affine_parts(model, x_coll)
        F, dF   = self._p_features_and_grads(model, K['x'])
        Phi = torch.cat([dF[:, :, 0], dF[:, :, 1]], 0).detach() * model.p_scale
        r   = b - self.mu_s * a                                   # (2N,1)
        G   = Phi.T @ Phi + lam_tik * torch.eye(Phi.shape[1], device=Phi.device, dtype=Phi.dtype)
        r_perp = r - Phi @ torch.linalg.solve(G, Phi.T @ r)
        scale = self.eta_0 * self.U_ref / (self.H_ref ** 2)       # adimensionalizzazione
        return (r_perp / scale).pow(2).mean()
```

> Nota di adimensionalizzazione: il residuo del momento ha scala $\eta_0 U/H^2$. Con $H=0.05$ e $\eta_0=1$, $U=1$: $\sim 400$. Senza divisione, $W_{mom}=1$ significa in realtà pesare il momento $\sim 1.6\times10^5$ volte rispetto a una loss dati $O(U^2)$. **Questo, da solo, spiega la distruzione della cinematica di Fase 1.**

### 5.5 Trust-region funzionale su $\psi$ + Lagrangiana aumentata

```python
class AugLagPhase2:
    r"""min L_mom^*  s.t.  L_data <= eps_d, L_bc <= eps_b, L_prox <= eps_psi
        Lagrangiana aumentata:  L = L_mom + sum_k [ mu_k*g_k + (rho_k/2) g_k^2 ]_+ """
    def __init__(self, eps, rho0=1.0, rho_max=1e4, mult0=1.0, gamma=2.0):
        self.eps = eps
        self.mult = {k: mult0 for k in eps}
        self.rho  = {k: rho0 for k in eps}
        self.gamma = gamma; self.rho_max = rho_max
        self.prev = {k: None for k in eps}

    def total(self, L_mom, terms):
        out = L_mom
        for k, Lk in terms.items():
            g = Lk / self.eps[k] - 1.0                     # violazione normalizzata
            g_plus = torch.clamp(g, min=0.0)
            out = out + self.mult[k] * g_plus + 0.5 * self.rho[k] * g_plus ** 2
        return out

    @torch.no_grad()
    def dual_update(self, terms):
        for k, Lk in terms.items():
            g = float(Lk) / self.eps[k] - 1.0
            self.mult[k] = max(0.0, self.mult[k] + self.rho[k] * max(0.0, g))
            if self.prev[k] is not None and max(0.0, g) > 0.5 * self.prev[k]:
                self.rho[k] = min(self.rho_max, self.gamma * self.rho[k])
            self.prev[k] = max(0.0, g)
```

Il termine proximal (in **spazio funzionale**, invariante per riparametrizzazione):

```python
def make_psi_reference(model, x_ref, physics):
    """Snapshot della cinematica di Fase 1 su una griglia fissa."""
    with torch.enable_grad():
        K = physics.kinematics2(model, x_ref.clone().requires_grad_(True))
    return torch.cat([K['u'], K['v']], 1).detach()

def loss_prox_psi(model, physics, x_ref, uv_ref):
    K = physics.kinematics2(model, x_ref.clone().requires_grad_(True))
    uv = torch.cat([K['u'], K['v']], 1)
    return ((uv - uv_ref) ** 2).mean() / (uv_ref.pow(2).mean() + 1e-12)
```

Scelta rigorosa delle soglie: $\epsilon_\psi=\big(\text{errore } L^2 \text{ relativo di Fase 1}\big)^2$, tipicamente $10^{-4}\!-\!10^{-6}$; $\epsilon_d=\sigma_{\text{dati}}^2$.

### 5.6 Loop di Fase 2 (sostituzione in `train.py`)

```python
def train_phase2(model, physics, data, x_coll, x_ref, cfg):
    # --- congelamenti ---
    for prm in model.model_tau.parameters(): prm.requires_grad_(False)
    physics.set_trainable('lam',  False)
    physics.set_trainable('mu_p', False)
    physics.set_trainable('mu_s', True)

    uv_ref = make_psi_reference(model, x_ref, physics)

    # gruppi con LR differenziati: psi si muove ~30x piu' lentamente di p
    opt = torch.optim.Adam([
        {'params': model.model_psi.parameters(), 'lr': cfg.lr_p / 30.0},
        {'params': model.model_p.parameters(),   'lr': cfg.lr_p},
        {'params': physics.get_phase2_params(),  'lr': cfg.lr_param},
    ])
    auglag = AugLagPhase2(eps={'data': cfg.eps_d, 'bc': cfg.eps_b, 'prox': cfg.eps_psi})

    # ---- VarPro iniziale: fornisce mu_s e p di ottima qualita' a costo zero ----
    diag = physics.varpro_pressure_and_mus(model, x_coll, update=True)
    print(f"[VarPro-0] mu_s={diag['mu_s']:.4f}  rho_id={diag['rho_id']:.2e}  "
          f"CRLB={diag['sd_mu_s']:.2e}")

    for it in tqdm(range(cfg.n_iter_p2)):
        opt.zero_grad(set_to_none=True)
        L_mom  = physics.loss_momentum_projected(model, x_coll)     # pressione eliminata
        L_data = physics.loss_data(model, data)
        L_bc   = physics.loss_bc_rolls(model)
        L_prox = loss_prox_psi(model, physics, x_ref, uv_ref)
        L = auglag.total(L_mom, {'data': L_data, 'bc': L_bc, 'prox': L_prox})
        L.backward()
        torch.nn.utils.clip_grad_norm_(
            [q for g in opt.param_groups for q in g['params']], 1.0)
        opt.step()

        if (it + 1) % cfg.dual_every == 0:
            auglag.dual_update({'data': L_data.detach(), 'bc': L_bc.detach(),
                                'prox': L_prox.detach()})
        # ---- passo di Gauss-Newton esatto sul blocco lineare ----
        if (it + 1) % cfg.varpro_every == 0:
            diag = physics.varpro_pressure_and_mus(
                model, x_coll, mu_s_prior=None, w_prior=0.0, update=True)

    # ================= FASE 2b: ricostruzione pressione (convessa) =================
    for prm in model.model_psi.parameters(): prm.requires_grad_(False)
    physics.set_trainable('mu_s', False)
    diag = physics.varpro_pressure_and_mus(model, x_coll_dense, update=True)  # closed form
    opt_p = torch.optim.LBFGS(model.model_p.parameters(), lr=1.0,
                              max_iter=cfg.lbfgs_iter, history_size=100,
                              line_search_fn='strong_wolfe',
                              tolerance_grad=1e-12, tolerance_change=1e-14)
    def closure():
        opt_p.zero_grad()
        L = physics.loss_pressure_recovery(model, x_coll_dense)
        L.backward(); return L
    opt_p.step(closure)
    return diag
```

con

```python
    def loss_pressure_recovery(self, model, x):
        r"""min_p || grad p - g ||^2,  g = -rho(u.grad)u + mu_s Lap u + div tau.
        Strettamente convessa modulo costante (fissata hard in model.pressure)."""
        b, a, K = self.momentum_affine_parts(model, x)
        g = -(b - self.mu_s.detach() * a)
        p = model.pressure(K['x'])                      # ancoraggio hard
        gp = self._grad(p, K['x'])
        gp = torch.cat([gp[:, 0:1], gp[:, 1:2]], 0)
        scale = self.eta_0 * self.U_ref / (self.H_ref ** 2)
        return ((gp - g.detach()) / scale).pow(2).mean()
```

### 5.7 Campionamento guidato dall'identificabilità (Optimal Experimental Design)

L'informazione è concentrata dove $|\mathbb P^\perp\Delta\mathbf u|$ è grande (zone di forte $\Delta\omega$: strati di taglio fra i rulli, intorno al punto di stagnazione iperbolico). Ricampionare con densità $\propto$ densità di informazione **massimizza $\mathcal I_{\text{eff}}$ a budget di punti fissato** (criterio D-ottimo per un singolo parametro):

```python
@torch.no_grad()
def resample_by_information(model, physics, x_pool, n_keep, alpha=1.0, floor=0.2):
    with torch.enable_grad():
        b, a, K = physics.momentum_affine_parts(model, x_pool.clone().requires_grad_(True))
        F, dF = physics._p_features_and_grads(model, K['x'])
    Phi = torch.cat([dF[:, :, 0], dF[:, :, 1]], 0) * model.p_scale
    G = Phi.T @ Phi + 1e-8 * torch.eye(Phi.shape[1], device=Phi.device)
    a_perp = (a - Phi @ torch.linalg.solve(G, Phi.T @ a)).abs()
    N = x_pool.shape[0]
    w = (a_perp[:N] ** 2 + a_perp[N:] ** 2).squeeze()
    w = w / w.mean()
    prob = (1 - floor) * (w ** alpha) / (w ** alpha).sum() + floor / N
    idx = torch.multinomial(prob, n_keep, replacement=False)
    return x_pool[idx]
```

### 5.8 Riparametrizzazione robusta in $\mu_{tot}$ (patch a `Physics`)

```python
    # --- in __init__ ---
    self.register_buffer("guess_mu_tot", torch.tensor(guess_mu_s + guess_mu_p,
                                                      device=DEVICE, dtype=torch.float32))
    self.register_parameter("_raw_mu_tot",
                            nn.Parameter(torch.zeros(1, device=DEVICE), requires_grad=False))
    self.use_mu_tot_param = False        # attivato in Fase 2

    @property
    def mu_tot_p2(self):
        return self.guess_mu_tot * torch.exp(self._raw_mu_tot).squeeze()

    @property
    def mu_s(self):
        if getattr(self, "use_mu_tot_param", False):
            # mu_p congelato da Fase 1; positivita' garantita in modo smooth
            return nn.functional.softplus(self.mu_tot_p2 - self.mu_p.detach(), beta=20.0)
        return self.guess_mu_s * torch.exp(self._raw_mu_s).squeeze()

    def get_phase2_params(self):
        if getattr(self, "use_mu_tot_param", False):
            return [self._raw_mu_tot]
        raw_p = getattr(self, "_raw_mu_s", None)
        return [raw_p] if isinstance(raw_p, nn.Parameter) else []
```

---

## 6. Perché questo sblocca la convergenza — riepilogo dei meccanismi

1. **Eliminazione esatta del gauge di pressione**: la loss vista dall'ottimizzatore è $\mathcal L^\star(\mu_s)=\tfrac12\|\mathbb P^\perp(\mathbf b-\mu_s\mathbf a)\|^2$, la cui Hessiana rispetto a $\mu_s$ è $\|\mathbb P^\perp\mathbf a\|^2$ — **strettamente positiva e nota a priori**. Nella formulazione (A) l'Hessiana ridotta è la stessa, ma l'ottimizzatore deve *scoprirla* navigando una valle con numero di condizione $\sim\rho_{id}^{-2}$ (tipicamente $10^{4}\!-\!10^{6}$): Adam non ci riesce, L-BFGS a malapena.
2. **Passo di Gauss–Newton esatto** sul blocco $(c,\mu_s)$: convergenza in *una* iterazione a $\psi$ fissato. Il residuo non lineare rimasto (solo $\theta_\psi$ e i layer nascosti di $p$) è molto più benigno.
3. **Adimensionalizzazione del residuo** per $\eta_0U/H^2$: elimina lo sbilanciamento $\sim10^5$ fra loss dati e loss momento che, con $W_{mom}=1$, garantiva la distruzione della cinematica di Fase 1.
4. **Trust-region funzionale + Lagrangiana aumentata**: la mobilità di $\psi$ diventa *controllata*; i pesi non sono più iperparametri arbitrari ma moltiplicatori determinati dalle soglie fisiche $\epsilon$.
5. **Diagnosi a priori**: $\rho_{id}$ e il CRLB vi dicono in $O(1)$ secondi se il vostro setup è informativo. Se non lo è, il rimedio non è algoritmico ma **sperimentale**.

---

## 7. Se $\rho_{id}$ risulta comunque troppo piccolo: aumentare l'informazione

Il Teorema 1 è una barriera **informazionale**, non numerica. Le vie per superarla, in ordine di efficacia:

**(a) Dati di trazione/coppia sui rulli.** Se COMSOL fornisce la coppia $M_k$ su ciascun rullo:
$$
M_k=\oint_{\Gamma_k}\big[\mathbf r\times\big(-p\mathbf n+2\mu_s\mathbf D\mathbf n+\boldsymbol\tau\mathbf n\big)\big]_z\,ds .
$$
La sensitività $\partial M_k/\partial\mu_s=2\oint(\mathbf r\times\mathbf D\mathbf n)_z ds$ è $O(1)$ e **non è annullata dalla pressione** (il contributo di $p$ alla coppia è quasi nullo per simmetria del rullo circolare). Quattro scalari con sensitività $O(1)$ superano milioni di punti di collocazione con sensitività $O(Re+Wi)$. **È la singola modifica con il maggior rapporto beneficio/costo.**

**(b) Condizione al contorno di trazione su una frontiera aperta** (se presente): $\big(-p\mathbf I+2\mu_s\mathbf D+\boldsymbol\tau\big)\mathbf n=\mathbf t_0$ accoppia direttamente $p$ e $\mu_s$ **senza** gauge libero, rimuovendo la degenerazione al bordo.

**(c) Multi-regime.** Dati a due velocità dei rulli $U_1\neq U_2$: $\boldsymbol\tau$ scala non linearmente con $Wi$ mentre $\mu_s\Delta\mathbf u$ scala linearmente ⇒ la collinearità $\mu_s\!\leftrightarrow\!\mu_p$ si rompe. Sensitività congiunta $\propto |Wi_1-Wi_2|$.

**(d) Fase 3 congiunta** con $\boldsymbol\tau$ sbloccato e prior gaussiano su $(\lambda,\mu_p)$ centrato sui valori di Fase 1 con varianza pari al CRLB di Fase 1 — l'unica procedura statisticamente coerente per propagare l'incertezza tra le fasi.

---

## 8. Note numeriche finali (non secondarie)

* **fp64 obbligatorio in Fase 2.** Le derivate seconde di $\psi$ in fp32 hanno errore relativo $\sim10^{-4}$; con $\rho_{id}\sim10^{-2}$ il segnale utile è $\rho_{id}\|\mathbf a\|\sim10^{-2}\|\mathbf a\|$ ⇒ SNR $\sim10^2$, al limite. In fp64 il problema sparisce. Usate `convert_to_fp64` **prima** di qualsiasi VarPro.
* **`ACTIVATION` = `tanh`** e non `SiLU`/`ReLU`: le derivate quarte (se optate per la forma (B)) o la regolarità di $\Delta\mathbf u$ richiedono $C^\infty$ con derivate limitate.
* **`p_scale=50`**: verificate che sia coerente con $\Delta p_{\text{fisico}}\sim\rho U^2+\eta_{tot}U/H$. Con VarPro l'ultimo layer viene risolto in closed-form, quindi `p_scale` diventa quasi irrilevante — ma influenza il condizionamento di $\Phi^\top\Phi$: normalizzate le feature ($F\leftarrow F/\|F\|_{\text{col}}$) prima del LS.
* **`torch.linalg.solve` sulle equazioni normali**: se $M\gtrsim256$, preferite `torch.linalg.lstsq(A, rhs, driver='gelsd')` sul sistema **non** normalizzato (condizionamento $\kappa$ invece di $\kappa^2$).
* **Monitoraggio**: tracciate a ogni VarPro `rho_id`, `sd_mu_s`, e la **norma di deriva cinematica** $\|\mathbf u-\mathbf u^{(1)}\|/\|\mathbf u^{(1)}\|$. Se quest'ultima supera l'1% mentre `L_mom` scende, siete nel regime di *gauge drift* descritto in §2.1 e dovete stringere $\epsilon_\psi$.

---

### Formula da tenere sotto gli occhi

$$
\boxed{\;
\frac{\operatorname{sd}(\hat\mu_s)}{\mu_s}\;\gtrsim\;\frac{\sigma_R}{\mu_s\,\big\|\mathbb P_{\mathcal H}\Delta\mathbf u\big\|}\;\sim\;\frac{\sigma_R\,H^3}{\mu_s\,U}\cdot\frac{1}{\mathcal O(Re+Wi)}\;+\;\frac{|\delta\mu_p|}{\mu_s}\;}
$$

Il primo termine è **varianza** (curabile con VarPro, fp64, campionamento ottimale, più punti); il secondo è **bias sistematico** da $\boldsymbol\tau$ congelato (curabile **solo** con riparametrizzazione in $\mu_{tot}$ o con la Fase 3 congiunta). Nel vostro setup con $\mu_s^{true}=0.1$ e $\mu_p^{true}=0.9$, un errore relativo dell'1% su $\mu_p$ produce un errore del **9%** su $\mu_s$: è il rapporto $\mu_p/\mu_s=9$ ad amplificare tutto. Riportate $\mu_{tot}$ come quantità primaria identificata e $\mu_s$ con la sua barra d'errore propagata.