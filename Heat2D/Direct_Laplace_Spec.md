# Setup Matematico e Dominio - Heat 2D Diretto (Laplace)

## 1. Setup Matematico

**Problema:** Conduzione di calore stazionaria in un dominio quadrato con condizioni al contorno miste (Laplace).

**Equazione di Laplace:**
$$
\nabla^2 T = \frac{\partial^2 T}{\partial x^2} + \frac{\partial^2 T}{\partial y^2} = 0
$$

**Condizioni al Contorno (Dirichlet):**
*   $T(x, 0) = 0$ (Lato inferiore)
*   $T(x, Ly) = 0$ (Lato superiore)
*   $T(0, y) = 0$ (Lato sinistro)
*   $T(Lx, y) = 1$ (Lato destro)

**Soluzione Analitica:**
La soluzione è data dalla serie:
$$
T(x,y) = \sum_{n=1,3,5...}^{\infty} \frac{4}{n\pi} \frac{\sinh(\lambda_n x)}{\sinh(\lambda_n L_x)} \sin(\lambda_n y), \quad \lambda_n = \frac{n\pi}{L_y}
$$
(Implementata troncando la somma ai primi $N=50$ termini).

## 2. Discretizzazione del Dominio

*   **Punti Dati (NN Random):** 1600 punti interni (Random) + 400 punti di bordo.
*   **Punti Dati (NN Grid):** 1600 punti interni (Griglia $40 \times 40$) + 400 punti di bordo.
*   **Punti Fisica (PINN Pure):** 1600 punti di collocazione (Griglia $40 \times 40$).
*   **Punti Dati (PINN Data+Phys):** 1000 punti (Subset Random) per supervisione dati + 1600 punti (Griglia) per fisica.
*   **Boundary:** 400 punti totali equidistanti (100 per lato).
*   **Griglia di Validazione:** $50 \times 50 = 2500$ punti.

---

## 3. Metriche di Validazione

### A. L2 Relative Error (Errore Globale)
Misura la distanza globale normalizzata tra predizione e verità, penalizzando gli outlier.

$$
Error_{L2} = \frac{\| T_{pred} - T_{true} \|_2}{\| T_{true} \|_2} = \sqrt{\frac{\sum_{i=1}^{N}(T_{pred}(i) - T_{true}(i))^2}{\sum_{i=1}^{N}(T_{true}(i))^2}}
$$

### B. Max Relative Error Peak (Errore Puntuale Massimo)
Indica la peggiore deviazione percentuale puntuale in zone significative. Utilizza una maschera $M$ per evitare singolarità dove $T$ è prossimo a zero.

**Definizione Maschera (Mask):**
$$
M = \{ i \mid |T_{true}(i)| > 0.01 \}
$$

**Formula:**
$$
PeakRel\% = \max_{i \in M} \left( \frac{|T_{pred}(i) - T_{true}(i)|}{|T_{true}(i)|} \right) \times 100
$$
