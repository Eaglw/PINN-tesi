# Handoff Sessione - PINN 4-Roll Mill: Riformulazione Fase 2 su Viscosità Totale ($\mu_{\text{tot}}$)

Data: 2026-09-02
Autore: Antigravity Assistant

## 1. Obiettivo Raggiunto
Completata la preparazione, verifica offline e smoke-test dello script **inal_roll/train_4roll_main_curl.py** per la **Fase 2 riformulata sulla Viscosità Totale $\mu_{\text{tot}}$** con vincolo di irrotazionalità del Momentum:
\text{curl}(\mathbf{F}) = \mu_{\text{tot}}^* \text{curl}(\nabla^2 \mathbf{u}) + \text{curl}\left( \nabla \cdot \boldsymbol{\tau}_E - Re_{\text{scale}}(\mathbf{u}\cdot\nabla)\mathbf{u} \right) \equiv 0
dove $\boldsymbol{\tau}_E = \boldsymbol{\tau} - 2\mu_p^{(F1)}\mathbf{D}(\mathbf{u})$ è lo stress puramente non-Newtoniano.

---

## 2. Risultati del Test Diagnostico Offline (su 25.000 punti)
Il confronto tra la formulazione standard su $\mu_s$ e la nuova su $\mu_{\text{tot}}$ ha evidenziato:
1. **Bilanciamento di Scala dei Termini**:
   - Formulazione standard ($\mu_s = 0.10$): $\frac{\text{RMS}(\nabla\cdot\boldsymbol{\tau})}{\text{RMS}(\mu_s \nabla^2 \mathbf{u})} = \mathbf{6.71}$ (il termine di stress sovrasta la diffusione di ~7×).
   - Nuova formulazione ($\mu_{\text{tot}} = 1.00$): $\frac{\text{RMS}(\nabla\cdot\boldsymbol{\tau}_E)}{\text{RMS}(\mu_{\text{tot}} \nabla^2 \mathbf{u})} = \mathbf{0.74}$ (termini in perfetto equilibrio d'ordine $\mathcal{O}(10^2)\text{ N/m}^3$).
2. **Segnale Non-Newtoniano $\boldsymbol{\tau}_E$**:
   - $\text{RMS}(|\boldsymbol{\tau}_E|) = 0.2891$ (pari al **.1\%$** dello stress totale, con correlazione $> 0.98$). Nessun collasso numerico per cancellazione.
3. **Condizionamento e Tolleranza al Rumore**:
   - L'incertezza relativa $\frac{\delta \mu_{\text{tot}}}{\mu_{\text{tot}}}$ è esattamente **\times$ più tollerante al rumore residuo di divergenza** rispetto a $\frac{\delta \mu_s}{\mu_s}$.

---

## 3. Stato dei Checkpoint & Pulizia
- **Checkpoint temporanei eliminati**: Rimossi da inal_roll/checkpoints/ i checkpoint intermedi delle prove precedenti.
- **Checkpoint di partenza conservato e verificato**:
  inal_roll/checkpoints/checkpoint_inverso_fase1_40k+10k.pth
  - $\lambda = 0.050203\text{ s}$ (Target: .050000$)
  - $\mu_p = 0.904854\text{ Pa}\cdot\text{s}$ (Target: .900000$)
- **Codice src/ intatto**: Nessun file in src/ è modificato (100% clean in git status).

---

## 4. Come Avviare lo Script sul PC di Maurizio

Script unico standalone:
**inal_roll/train_4roll_main_curl.py**

### Configurazione del Training:
- **Punto di Partenza**: checkpoint_inverso_fase1_40k+10k.pth (Fase 1 completata).
- **Fase 2 Adam**: 30.000 epoche (epoche 50.001 $\to$ 80.000) su $\mu_{\text{tot}}$ e pressione $.
- **Fase 2 L-BFGS**: 2.000 iterazioni in FP64 su $ e $\mu_{\text{tot}}$.
- **Parametro Ottimizzato**: $\mu_{\text{tot}}$ (guess: .000\text{ Pa}\cdot\text{s}$, target: .000\text{ Pa}\cdot\text{s}$).
- **Parametro Derivato**: $\mu_s = \mu_{\text{tot}} - \mu_p^{(F1)}$ (target: .100\text{ Pa}\cdot\text{s}$).
- **Vincolo Rotazionale**: {\text{curl}} = 1.0$ valutato su 5.000 punti con precalcolo di '$.

### Comando di Avvio da PowerShell:
`powershell
.\venv\Scripts\python final_roll/train_4roll_main_curl.py
`
*(Oppure python final_roll/train_4roll_main_curl.py a seconda del setup dell'interprete).*

Lo smoke-test di prova ha confermato il corretto avvio, la discesa della loss e la totale indipendenza da file esterni.
