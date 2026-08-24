---
date: 2026-08-21T18:28:19
inverse_problem: True
dataset: 4_roll_mill.csv
epochs: 35000
---

# Run_011_4_roll_mill_L8x128_E35000_SiLU_stagedTrue_invTrue_20260821_182817

## 💡 Idea
Scrivi qui l'obiettivo o la descrizione di questa run...

## 📝 Dettagli Configurazione
- **dataset**: 4_roll_mill.csv
- **epochs**: 35000
- **inverse_problem**: True
- **staged_training**: True
- **activation**: SiLU
- **network**: 8x128
- **lbfgs_phase1**: True
- **lbfgs_phase2**: True

## 📊 Risultati Finali

### 🔹 Parametri e Numeri Adimensionali
| Parametro | Valore |
|---|---|
| **beta** | 0.460738 |
| **mu_s** | 1.600959 |
| **mu_p** | 1.873810 |
| **lam** | 0.051855 |
| **eps** | 0.000000 |
| **alpha** | 0.000000 |

### 🔹 Metriche di Loss
| Metrica | Valore |
|---|---|
| **Data Loss** | 8.246995e-05 |
| **Boundary Loss** | 1.048506e-04 |
| **BC_u** | 4.588933e-05 |
| **BC_v** | 5.688980e-05 |
| **BC_p** | 1.309146e-09 |
| **BC_tau** | 2.070164e-06 |
| **Momentum Loss** | 1.797117e-03 |
| **Constitutive Loss** | 1.838499e-03 |
| **Total PDE Loss** | 3.635616e-03 |
| **Total Loss** | 1.151357e-02 |
| **Mean Abs f_u** | 2.335501e-02 |
| **Mean Abs f_v** | 2.462570e-02 |
| **Mean Abs f_txx** | 3.472590e-02 |
| **Mean Abs f_txy** | 3.200100e-02 |
| **Mean Abs f_tyy** | 3.463037e-02 |

### 🔹 Errori Relativi L2
| Variabile | Errore L2 |
|---|---|
| **u** | 0.008889 |
| **v** | 0.009270 |
| **p** | 0.790833 |
| **tau_xx** | 0.037412 |
| **tau_xy** | 0.027823 |
| **tau_yy** | 0.040104 |
| **tau_xx_masked** | 0.037375 |
| **tau_xy_masked** | 0.027750 |
| **tau_yy_masked** | 0.040080 |

## 🖼️ Plot Generati

### Global Fields
![[global_fields.png]]

### High Stress
![[high_stress.png]]

### L2 Errors History
![[l2_errors_history.png]]

### Loss History
![[loss_history.png]]

### Params Evolution
![[params_evolution.png]]

## 📜 Log della Run
- [[train_log.txt]]

## 📌 Note e Conclusioni
- 
