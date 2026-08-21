---
date: 2026-08-18T20:05:41
inverse_problem: True
dataset: 4_roll_mill.csv
epochs: 35000
---

# Run_010_4_roll_mill_L8x128_E35000_SiLU_stagedTrue_invTrue_20260818_200540

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
| **beta** | 0.000000 |
| **mu_s** | 0.000000 |
| **mu_p** | 0.027006 |
| **lam** | 0.050755 |
| **eps** | 0.000000 |
| **alpha** | 0.000000 |

### 🔹 Metriche di Loss
| Metrica | Valore |
|---|---|
| **Data Loss** | 4.363583e-05 |
| **Boundary Loss** | 1.981591e-05 |
| **BC_u** | 8.331234e-06 |
| **BC_v** | 8.304360e-06 |
| **BC_p** | 1.369052e-10 |
| **BC_tau** | 3.180177e-06 |
| **Momentum Loss** | 1.963063e-01 |
| **Constitutive Loss** | 2.071459e-03 |
| **Total PDE Loss** | 1.983777e-01 |
| **Total Loss** | 5.952760e-01 |
| **Mean Abs f_u** | 3.618096e-01 |
| **Mean Abs f_v** | 3.584201e-01 |
| **Mean Abs f_txx** | 4.030723e-02 |
| **Mean Abs f_txy** | 4.087566e-02 |
| **Mean Abs f_tyy** | 3.982365e-02 |

### 🔹 Errori Relativi L2
| Variabile | Errore L2 |
|---|---|
| **u** | 0.006670 |
| **v** | 0.006541 |
| **p** | 2.587743 |
| **tau_xx** | 0.026733 |
| **tau_xy** | 0.025866 |
| **tau_yy** | 0.021705 |
| **tau_xx_masked** | 0.026659 |
| **tau_xy_masked** | 0.025755 |
| **tau_yy_masked** | 0.021636 |

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
