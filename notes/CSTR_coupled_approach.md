# Approccio Coupled PINN per CSTR Non-Isotermo

## Ispirazione: ViscoelasticNet
Basandosi sull'approccio descritto nel paper "ViscoelasticNet", separiamo le variabili fisiche che hanno dinamiche e scale diverse in reti neurali distinte, accoppiandole tramite la funzione di loss fisica.

## Modello Fisico
Il sistema CSTR non è più considerato isotermo. Abbiamo un sistema di due equazioni differenziali ordinarie (ODE) accoppiate.

### 1. Bilancio di Massa (Concentrazione $C_A$)
$$ \frac{dC_A}{dt} = \frac{F}{V}(C_{in} - C_A) - k(T) C_A $$

### 2. Bilancio di Energia (Temperatura $T$)
$$ \frac{dT}{dt} = \frac{F}{V}(T_{in} - T) - \frac{\Delta H}{\rho C_p} k(T) C_A + \frac{UA}{\rho C_p V}(T_{cool} - T) $$

### 3. Cinetica di Reazione (Arrhenius)
Il coefficiente di velocità $k$ dipende dalla temperatura:
$$ k(T) = k_0 \exp\](-\frac{E}{R T}\]) $$

## Architettura Multi-Network
Invece di una singola rete `PINN(t) -> [C, T]`, utilizziamo due reti specializzate:

1.  **ConcentrationNet ($N_C$):**
    *   Input: Tempo $t$
    *   Output: Concentrazione $C_A$
    *   Obiettivo: Catturare la dinamica di consumo del reagente.

2.  **TemperatureNet ($N_T$):**
    *   Input: Tempo $t$
    *   Output: Temperatura $T$
    *   Obiettivo: Catturare il riscaldamento dovuto alla reazione esotermica e il raffreddamento.

## Loss Function Accoppiata
La loss totale è la somma delle loss sui dati (se disponibili) e dei residui fisici incrociati.

$$ \mathcal{L}_{tot} = \mathcal{L}_{data,C} + \mathcal{L}_{data,T} + \lambda_1 \mathcal{L}_{ODE, Mass} + \lambda_2 \mathcal{L}_{ODE, Energy} $$

Dove i residui fisici sono calcolati usando `torch.autograd` e incrociando gli output delle due reti: il termine di reazione $k(N_T(t)) \cdot N_C(t)$ appare in entrambe le equazioni, fungendo da "collante".

## Note Implementative
*   **Generazione Dati:** Poiché la soluzione analitica del main è solo isoterma, questo modulo deve generare i propri dati di "ground truth" numericamente (es. RK4) per il training/validazione.
*   **Parametri:** Sono stati introdotti nuovi parametri termici ($\Delta H, E, UA, \dots$) necessari per il bilancio energetico.
