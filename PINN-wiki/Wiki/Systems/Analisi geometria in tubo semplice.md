# PINN Viscoelastico — Perché l'Inverse Problem Fallisce in Canale Rettangolare

> **Contesto**: Analisi matematica completa del perché un problema inverso PINN per fluidi Giesekus/Oldroyd-B non può convergere in geometria rettangolare con flusso stazionario pienamente sviluppato (Poiseuille flow). Documento generato dalla sessione di debugging del 28 maggio 2026.

---

## 1. Il Setup del Problema

### 1.1 Architettura della PINN

Il sistema risolve le equazioni di Navier-Stokes accoppiate al modello costitutivo Giesekus/PTT unificato. La rete neurale predice sei campi:

$$\mathcal{N}(x, y; \theta) \rightarrow [\psi, p, \tau_{xx}, \tau_{xy}, \tau_{yy}]$$

dove la stream function $\psi$ garantisce automaticamente $\nabla \cdot \mathbf{u} = 0$ tramite:

$$u = \frac{\partial \psi}{\partial y}, \quad v = -\frac{\partial \psi}{\partial x}$$

### 1.2 Equazioni Fisiche

**Equazioni di Quantità di Moto (Navier-Stokes):**

$$\rho\left(u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y}\right) + \frac{\partial p}{\partial x} - \mu_s \nabla^2 u - \frac{\partial \tau_{xx}}{\partial x} - \frac{\partial \tau_{xy}}{\partial y} = 0 \tag{1}$$

$$\rho\left(u \frac{\partial v}{\partial x} + v \frac{\partial v}{\partial y}\right) + \frac{\partial p}{\partial y} - \mu_s \nabla^2 v - \frac{\partial \tau_{xy}}{\partial x} - \frac{\partial \tau_{yy}}{\partial y} = 0 \tag{2}$$

**Modello Costitutivo Giesekus (componente $\tau_{xx}$):**

$$\tau_{xx} + \lambda \overset{\nabla}{\tau}_{xx} + \frac{\alpha\lambda}{\mu_p}(\tau_{xx}^2 + \tau_{xy}^2) = 2\mu_p \frac{\partial u}{\partial x} \tag{3}$$

**Modello Costitutivo Giesekus (componente $\tau_{xy}$):**

$$\tau_{xy} + \lambda \overset{\nabla}{\tau}_{xy} + \frac{\alpha\lambda}{\mu_p}\tau_{xy}(\tau_{xx} + \tau_{yy}) = \mu_p\left(\frac{\partial u}{\partial y} + \frac{\partial v}{\partial x}\right) \tag{4}$$

**Modello Costitutivo Giesekus (componente $\tau_{yy}$):**

$$\tau_{yy} + \lambda \overset{\nabla}{\tau}_{yy} + \frac{\alpha\lambda}{\mu_p}(\tau_{xy}^2 + \tau_{yy}^2) = 2\mu_p \frac{\partial v}{\partial y} \tag{5}$$

dove $\overset{\nabla}{\tau}$ è la derivata upper-convected:

$$\overset{\nabla}{\tau}_{ij} = \frac{D\tau_{ij}}{Dt} - (\nabla \mathbf{u}) \cdot \tau - \tau \cdot (\nabla \mathbf{u})^T \tag{6}$$

### 1.3 Parametri da Identificare (Inverse Mode)

Nel problema inverso, i parametri $\{\mu_s, \mu_p, \lambda, \alpha\}$ sono `nn.Parameter` addestrabili. La loss totale diventa:

$$\mathcal{L}_{total} = w_m \cdot \mathcal{L}_{momentum} + w_c \cdot \mathcal{L}_{constitutive} + \mathcal{L}_{BC} \tag{7}$$

---

## 2. Il Paradosso dell'Identificabilità: Analisi Matematica

### 2.1 Semplificazione delle PDE in Poiseuille Stazionario Pienamente Sviluppato

In un canale rettangolare con flusso stazionario pienamente sviluppato, valgono per definizione le seguenti condizioni:

$$\frac{\partial(\cdot)}{\partial x} = 0, \quad v = 0, \quad \frac{\partial(\cdot)}{\partial t} = 0 \tag{8}$$

**Effetto sulla derivata upper-convected:** Sostituendo (8) in (6), tutti i termini di trasporto si annullano:

$$\overset{\nabla}{\tau}_{ij} = \underbrace{u \frac{\partial \tau_{ij}}{\partial x}}_{=0} + \underbrace{v \frac{\partial \tau_{ij}}{\partial y}}_{=0} - (\nabla \mathbf{u})\cdot\tau - \tau\cdot(\nabla\mathbf{u})^T$$

Poiché l'unico gradiente di velocità non nullo è $\dot{\gamma} = \partial u / \partial y$, la derivata upper-convected si riduce a:

$$\overset{\nabla}{\tau}_{xx} = -2\dot{\gamma}\tau_{xy}, \quad \overset{\nabla}{\tau}_{xy} = -\dot{\gamma}\tau_{yy}, \quad \overset{\nabla}{\tau}_{yy} = 0 \tag{9}$$

### 2.2 Il Sistema Diventa Algebrico

Sostituendo (9) nelle equazioni (3)-(5), il sistema costitutivo si trasforma in un **sistema algebrico non lineare**:

$$\tau_{xx} - 2\lambda\dot{\gamma}\tau_{xy} + \frac{\alpha\lambda}{\mu_p}(\tau_{xx}^2 + \tau_{xy}^2) = 0 \tag{10}$$

$$\tau_{xy} - \lambda\dot{\gamma}\tau_{yy} + \frac{\alpha\lambda}{\mu_p}\tau_{xy}(\tau_{xx} + \tau_{yy}) = \mu_p\dot{\gamma} \tag{11}$$

$$\tau_{yy} + \frac{\alpha\lambda}{\mu_p}(\tau_{xy}^2 + \tau_{yy}^2) = 0 \tag{12}$$

**Questo è il nucleo del problema**: le PDE alle derivate parziali si sono ridotte a equazioni algebriche. Non c'è più propagazione spaziale — ogni punto del dominio è indipendente dagli altri lungo x.

### 2.3 Analisi dei Gradi di Libertà

Il sistema (10)-(12) ha:
- **3 equazioni** (una per componente dello stress)
- **Incognite parametriche**: $\mu_s, \mu_p, \lambda, \alpha$ → **4 parametri**

Il sistema è **sottovincolato di 1 grado di libertà**. Esiste una famiglia continua di soluzioni $\{\mu_s^*, \mu_p^*, \lambda^*, \alpha^*\}$ che soddisfano simultaneamente le equazioni (10)-(12) per qualsiasi dato di velocità.

### 2.4 Perché il Profilo di Velocità Non Aiuta

L'equazione del moto (1) in regime pienamente sviluppato diventa:

$$\frac{dp}{dx} = \frac{\partial \tau_{xy}}{\partial y} + \mu_s \frac{\partial^2 u}{\partial y^2} \tag{13}$$

Questa equazione vincola **solo la somma** $\mu_s + \mu_p$ (la viscosità totale), non la loro suddivisione. Il profilo parabolico $u(y)$ è identico per qualsiasi coppia $(\mu_s, \mu_p)$ tale che $\mu_s + \mu_p = \mu_{total}$.

**Dimostrazione formale**: La soluzione analitica per il profilo di velocità in Poiseuille viscoelastico è:

$$u(y) = \frac{-\nabla p}{2(\mu_s + \mu_p)} y(L_y - y) \tag{14}$$

Il parametro $\lambda$ non compare in (14). Infiniti valori di $\lambda$ generano la **stessa identica parabola**.

### 2.5 Il Gradiente della Loss è Piatto rispetto a λ

Formalmente, il gradiente che aggiorna $\lambda$ durante il training è:

$$\frac{\partial \mathcal{L}}{\partial \lambda} = \sum_k \frac{\partial \mathcal{L}}{\partial f_k} \cdot \frac{\partial f_k}{\partial \lambda} \tag{15}$$

dove $f_k \in \{f_u, f_v, f_{\tau_{xx}}, f_{\tau_{xy}}, f_{\tau_{yy}}\}$. In flusso pienamente sviluppato:

- $\partial f_u / \partial \lambda = 0$ (λ non appare nell'equazione del moto 1D)
- $\partial f_v / \partial \lambda = 0$ (idem)
- $\partial f_{\tau}/\partial \lambda$ dipende dai termini $\lambda \cdot \overset{\nabla}{\tau}$, ma questi sono **già zero** per (8)

Rimane solo il termine algebrico $\alpha\lambda/\mu_p \cdot (\tau_{ij}^2)$, ma questo crea una **superficie di loss piatta** perché molteplici coppie $(\lambda, \alpha)$ producono lo stesso residuo.

---

## 3. Perché le Boundary Conditions dell'Inlet Non Risolvono il Problema

### 3.1 Cosa Fornisce l'Inlet

Il codice impone all'inlet ($x=0$):

```
Dirichlet: u(0,y), v=0, p(0,y), τ_xx(0,y), τ_xy(0,y), τ_yy(0,y)
```

Questi valori rappresentano la **soluzione esatta del sistema algebrico (10)-(12)** per i parametri reali. Fornirli alla PINN equivale a dare la risposta al modello per $x=0$.

### 3.2 Il Problema della Ridondanza

In flusso pienamente sviluppato, la soluzione è **costante lungo x**:

$$\tau_{ij}(x, y) = \tau_{ij}(0, y) \quad \forall x \in [0, L_x] \tag{16}$$

Quindi dare i valori all'inlet equivale a dare la soluzione **in tutto il dominio**. Ma questo non aiuta l'identificazione dei parametri, perché lo stesso set di valori $\{\tau_{xx}(y), \tau_{xy}(y), \tau_{yy}(y)\}$ è compatibile con infinite coppie $(\lambda, \alpha, \mu_p)$.

### 3.3 La Condizione Zero-Gradient all'Outlet è Ridondante

Il codice impone all'outlet:

```
Neumann: ∂τ_xx/∂n = 0, ∂τ_xy/∂n = 0
```

Ma per (8), questa condizione è già soddisfatta **per costruzione matematica** del problema. Non aggiunge nessuna informazione indipendente.

### 3.4 Analisi della Matrice di Informazione di Fisher

L'identificabilità di un parametro $\theta_i$ è legata alla matrice di Fisher:

$$F_{ij} = \mathbb{E}\left[\frac{\partial \log p(\mathbf{u}|\theta)}{\partial \theta_i} \frac{\partial \log p(\mathbf{u}|\theta)}{\partial \theta_j}\right] \tag{17}$$

Se $\partial \mathbf{u}/\partial \lambda = 0$ (il profilo di velocità non dipende da $\lambda$), la riga e colonna corrispondente a $\lambda$ in $F$ sono **identicamente zero**. La matrice è singolare, e la stima di massima verosimiglianza è indefinita — il problema inverso è **mal posto** nel senso di Hadamard.

---

## 4. Il Problema con `torch.abs` sui Parametri

Il codice usa:

```python
lam_eff  = torch.abs(self.lam)
alpha_eff = torch.abs(self.alpha)
```

La funzione $|x|$ ha **gradiente zero in $x=0$** e **gradiente discontinuo** per tutti gli $x$. Se il parametro attraversa zero durante il training, il gradiente si annulla esattamente e l'ottimizzatore si blocca.

**Soluzione corretta**: usare `softplus`:

$$\text{softplus}(x) = \frac{1}{\beta}\log(1 + e^{\beta x}) \tag{18}$$

che è ovunque smooth, ovunque positiva, e ha gradiente $\sigma(\beta x)$ (sigmoide) sempre non-nullo.

```python
lam_eff   = torch.nn.functional.softplus(self.lam)
alpha_eff = torch.nn.functional.softplus(self.alpha)
```

---

## 5. Quando il Problema Inverso Funziona

### 5.1 Condizioni Necessarie per l'Identificabilità

Perché $\lambda$ sia identificabile dalla sola velocità, è necessario che il flusso presenti regioni di **elongazione** dove:

$$\frac{\partial u}{\partial x} \neq 0 \tag{19}$$

In questi casi i termini di trasporto nella derivata upper-convected sono non-nulli, e il profilo di velocità dipende fortemente da $\lambda$.

### 5.2 Geometrie Adatte vs Non Adatte

| Geometria | $\partial u/\partial x$ | λ identificabile da u? | Note |
|---|---|---|---|
| Canale rettangolare dritto | = 0 | ❌ No | Problema della presente analisi |
| Canale convergente (>5°) | ≠ 0 | ⚠️ Debole | Migliora con angolo maggiore |
| Stenosi (contrazione-espansione) | ≠ 0 forte | ✅ Sì | Usato nei paper di riferimento |
| Cross-slot (flusso a croce) | ≠ 0 forte | ✅ Sì | Punto di stagnazione elongazionale |
| Canale ondulato | ≠ 0 periodico | ✅ Sì | Alternativa computazionalmente efficiente |

### 5.3 Cosa Identifica Cosa

| Dati forniti | μ_total | μ_s/μ_p | λ | α |
|---|---|---|---|---|
| Solo u(y) in canale dritto | ✅ | ❌ | ❌ | ❌ |
| u(y) + τ_xy all'inlet | ✅ | ⚠️ parziale | ❌ | ❌ |
| u(y) + τ_xy + τ_xx all'inlet | ✅ | ✅ | ⚠️ debole | ⚠️ debole |
| u(y) + τ interno (anche pochi punti) | ✅ | ✅ | ✅ | ✅ |
| u(x,y) in geometria complessa | ✅ | ✅ | ✅ | ✅ |

---

## 6. Soluzioni Pratiche

### 6.1 Opzione A — Supervisione su Stress Interni

Aggiungere una loss di dati su punti interni del dominio:

```python
def data_loss(self, model, x_data, tau_data):
    u, v, p, tau = self.get_velocity(model, x_data)
    return F.mse_loss(tau, tau_data)
```

Anche 10-20 punti interni con $\tau_{xx}$ esatto sono sufficienti per rompere la degenerazione.

### 6.2 Opzione B — Estrarre i Parametri dal Dataset (Refactoring Goal 0)

Se il dataset contiene la soluzione analitica, i parametri possono essere estratti **direttamente** dalle equazioni algebriche (10)-(12) senza training inverso. Questo è il refactoring corretto per Goal 0.

### 6.3 Opzione C — Geometria Non Banale

Introdurre anche una piccola perturbazione geometrica (canale leggermente convergente, singola stenosi lieve) attiva i termini di trasporto e rende il problema ben posto.

---

## 7. Sommario delle Cause di Fallimento

| Causa | Tipo | Gravità | Soluzione |
|---|---|---|---|
| Flusso pienamente sviluppato → PDE algebrica | Strutturale | 🔴 Critica | Cambia geometria o aggiungi dati τ interni |
| Profilo u(y) indipendente da λ | Matematica | 🔴 Critica | Vedi sopra |
| Outlet Neumann ridondante | Ridondanza | 🟡 Lieve | Nessun impatto se le altre BC sono corrette |
| `torch.abs` → gradiente nullo in 0 | Implementazione | 🟠 Media | Sostituire con `softplus` |
| Sistema algebrico sottovincolato (3 eq, 4 param) | Matematica | 🔴 Critica | Aggiungere vincolo indipendente |

---

*Documento generato il 28 maggio 2026 — sessione di analisi PINN viscoelastico.*
