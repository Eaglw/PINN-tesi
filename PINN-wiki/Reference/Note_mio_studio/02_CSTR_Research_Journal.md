# CSTR Research Journal: Modeling & Optimization

Questo giornale documenta la ricerca, la pianificazione e i risultati relativi alla modellazione di un reattore Continuo Stirred-Tank (CSTR) tramite PINNs.

---

## 1. Stato del Progetto e Obiettivi
Le attività sul CSTR si sono evolute da una fase di pianificazione a un framework di sperimentazione sistematica, includendo:
- **Framework Automatizzato**: Ciclo di esecuzione per grid search su architetture e ottimizzatori.
- **Modello Flessibile**: Classe `FCN` configurabile per diverse funzioni di attivazione.
- **Confronto Ottimizzatori**: Test su Adam, LBFGS e approcci ibridi (Adam -> LBFGS).
- **Varietà di Scenari**: Supporto per problemi diretti (con e senza dati) e problemi inversi (stima parametri).

---

## 2. Analisi Comparativa degli Ottimizzatori e Attivazioni

### 2.1 Strategie di Ottimizzazione
- **Adam**: Baseline robusta per la convergenza iniziale.
- **LBFGS**: Metodo quasi-Newtoniano per alta precisione finale (richiede la `closure` per il calcolo della loss).
- **Approccio Ibrido**: Risultata la strategia più potente. Adam per N epoche per avvicinarsi al minimo, poi LBFGS per la rifinitura.

### 2.2 Piano Sperimentale Combinato
| Ottimizzatore | Funzione di Attivazione | Note |
| :--- | :--- | :--- |
| Adam | Tanh | Baseline di riferimento |
| Adam -> LBFGS | GELU | Combinazione avanzata (promettente) |

---

## 3. Approccio Coupled PINN (CSTR Non-Isotermo)

Per sistemi più complessi (non isotermi), è stato implementato un approccio **Coupled PINN** ispirato a *ViscoelasticNet*.

### 3.1 Architettura Multi-Network
Invece di una singola rete, vengono utilizzate due reti specializzate accoppiate dalla fisica:
1.  **ConcentrationNet ($N_C$):** Predice la concentrazione $C_A$.
2.  **TemperatureNet ($N_T$):** Predice la temperatura $T$.

### 3.2 Loss Function Accoppiata
La loss include i residui fisici incrociati dei bilanci di massa ed energia:
- **Bilancio Massa:** $\frac{dC_A}{dt} = \dots - k(T) C_A$
- **Bilancio Energia:** $\frac{dT}{dt} = \dots - \Delta H \cdot k(T) C_A + \dots$
Il termine di Arrhenius $k(T)$ funge da "collante" tra le due reti.

### 3.3 Sfide Implementative e Soluzioni
- **Esplosione del Gradiente (Arrhenius):** Temperature iniziali errate possono causare overflow numerici nell'esponenziale.
    - **Soluzione:** Strategia di **Warm-up**. Nei primi 1000 step, la loss fisica è nulla ($\lambda_{phys} = 0$). La rete impara solo dai dati per stabilizzarsi in un range termico sensato prima di attivare i vincoli fisici.
- **Tuning dei Parametri:** Necessità di adattare $k_0$ per evitare dinamiche troppo veloci rispetto alla finestra temporale osservata (0-2s).
