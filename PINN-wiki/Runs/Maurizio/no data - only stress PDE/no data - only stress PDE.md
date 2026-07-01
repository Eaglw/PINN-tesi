---
date: 2026-06-22T10:11:28
inverse_problem: False
dataset: 4_roll_mill.csv
epochs: 100000
---

# Run_002_4_roll_mill_L8x128_E100000_SiLU_stagedTrue_invFalse_20260621_174646

## 📝 Dettagli Configurazione
- **dataset**: 4_roll_mill.csv
- **epochs**: 100000
- **inverse_problem**: False
- **staged_training**: True
- **activation**: SiLU
- **network**: 8x128
- **lbfgs**: False

## 📊 Risultati Finali

### 🔹 Parametri e Numeri Adimensionali
| Parametro | Valore |
|---|---|
| **mu_s** | 0.100000 |
| **mu_p** | 0.900000 |
| **lam** | 0.050000 |
| **eps** | 0.000000 |
| **alpha** | 0.000000 |

### 🔹 Metriche di Loss
| Metrica | Valore |
|---|---|
| **Data Loss** | 2.329688e-02 |
| **Boundary Loss** | 1.045693e+00 |
| **BC_u** | 1.877478e-06 |
| **BC_v** | 1.820929e-06 |
| **BC_p** | 1.045689e+00 |
| **Momentum Loss** | 7.888380e+02 |
| **Constitutive Loss** | 1.164042e-07 |
| **Total PDE Loss** | 7.888380e+02 |
| **Total Loss** | 2.368605e+03 |
| **Mean Abs f_u** | 2.488640e+01 |
| **Mean Abs f_v** | 2.494687e+01 |
| **Mean Abs f_txx** | 2.514458e-04 |
| **Mean Abs f_txy** | 2.152428e-04 |
| **Mean Abs f_tyy** | 2.534663e-04 |

### 🔹 Errori Relativi L2
| Variabile | Errore L2 |
|---|---|
| **u** | 0.152629 |
| **v** | 0.152636 |
| **p** | 1.000000 |
| **tau_xx** | 0.796674 |
| **tau_xy** | 0.782680 |
| **tau_yy** | 0.791279 |
| **tau_xx_masked** | 0.796672 |
| **tau_xy_masked** | 0.782679 |
| **tau_yy_masked** | 0.791277 |

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
