# Machine Learning in Chemical Engineering & Rheology

*Note basate sulla ricerca del Prof. **Safa Jamali** (Northeastern University) e integrato con la letteratura scientifica recente.*

Este documento serve come approfondimento teorico sul ruolo del ML nella reologia e nei fluidi complessi (RhINNs).

---

## 1. I Quattro Pilastri del ML nell'Ingegneria Chimica

1.  **Flow Modeling:** Uso di reti neurali (RhINNs) per integrare bilanci di massa e quantità di moto nella loss function, permettendo di simulare fluidi non-newtoniani con pochi dati.
2.  **Automated Experiment Data:** Integrazione di robotica e ML per esperimenti *high-throughput*. Uso di *Active Learning* per decidere il prossimo esperimento ottimale.
3.  **Material Modeling & Characterization:** Mappatura della microstruttura (es. distribuzione particelle) sulle proprietà macroscopiche (viscosità, elasticità).
4.  **Model Discovery:** Uso del ML per scoprire l'equazione differenziale che governa il sistema (es. tecniche SINDy) invece di assumerne una a priori.

---

## 2. Concetti Chiave

### Teoria della Fluidità Locale
Legame tra microscopic state (Granular Packing $\phi$) e proprietà macroscopiche. La "fluidità" ($g$) è definita come l'inverso della viscosità locale e risente del "jamming" quando il packing è troppo alto.

### L'importanza della Conoscenza Fisica ("Baffi dei Gatti")
Esempio concettuale: classificare immagini di cani/gatti richiede $10^6$ campioni. Se imponiamo vincoli fisici (es. equazioni che descrivono i baffi), il numero di campioni necessari scende drasticamente a $100$. Questo è il potere delle PINNs.

---

## 3. Multi-Fidelity Neural Networks (MFNN)

Strategia a due stadi per ridurre i costi:
-   **Rete Low-Fidelity (LF):** Addestrata su dataset enormi ma approssimativi (simulazioni veloci). Impara il **trend** generale.
-   **Rete High-Fidelity (HF):** Addestrata su pochi dati sperimentali scarsi e costosi. Prende l'output della LF e impara solo la **correzione** (l'errore del modello semplificato).

---

## 4. Direzioni di Sviluppo

-   **Generalizzazione Geometrica:** Uso di *Neural Operators* (DeepONet, FNO) o *Graph Neural Networks* (GNN) per rendere il modello agnostico rispetto alla geometria del condotto.
-   **Gestione della Memoria (Viscoelasticità):** Uso di LSTM o derivate frazionarie per catturare l'effetto memoria dei fluidi dove lo stress attuale dipende dalla storia della deformazione.
