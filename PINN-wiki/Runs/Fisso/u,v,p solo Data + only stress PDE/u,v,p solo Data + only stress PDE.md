---
date: 2026-06-24T00:54:09
inverse_problem: false
dataset: 4_roll_mill.csv
epochs: 88000
Computer: Fisso
staged: true
inverse: false
---

# Run_001_4_roll_mill_StressOnly_L8x128_E1=50000_E2=30000_SiLU_invFalse_20260623_163555

## 💡 Idea
Esattamente uguale a [[u,v,p solo Data - solo PDE stress - wrong lambda]] ma fixando il problema del lambda. Forse dovrei rivedere anche [[u,v,p solo data 50+5k - entrambe PDE 30+3k]], ma sono fiducioso che il training contemporaneo sia problematico. 

## 📝 Dettagli Configurazione
- **dataset**: 4_roll_mill.csv
- **epochs_phase1**: 50000
- **epochs_phase2**: 30000
- **inverse_problem**: False
- **staged_training**: False
- **activation**: SiLU
- **network**: 8x128
- **lbfgs**: True

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
| **Data Loss** | 1.458232e-06 |
| **Boundary Loss** | 8.995587e-05 |
| **BC_u** | 4.901366e-05 |
| **BC_v** | 3.383864e-05 |
| **BC_p** | 7.103574e-06 |
| **Momentum Loss** | 9.892979e+01 |
| **Constitutive Loss** | 1.009063e-06 |
| **Total PDE Loss** | 9.892979e+01 |
| **Total Loss** | 2.967895e+02 |
| **Mean Abs f_u** | 7.085975e+00 |
| **Mean Abs f_v** | 6.884987e+00 |
| **Mean Abs f_txx** | 6.209393e-04 |
| **Mean Abs f_txy** | 5.101333e-04 |
| **Mean Abs f_tyy** | 5.602966e-04 |

### 🔹 Errori Relativi L2
| Variabile | Errore L2 |
|---|---|
| **u** | 0.001211 |
| **v** | 0.001205 |
| **p** | 0.002821 |
| **tau_xx** | 0.023608 |
| **tau_xy** | 0.019978 |
| **tau_yy** | 0.020105 |
| **tau_xx_masked** | 0.023605 |
| **tau_xy_masked** | 0.019968 |
| **tau_yy_masked** | 0.020100 |

## 🖼️ Plot Generati

### Global Fields
![[Runs/Fisso/u,v,p solo Data + only stress PDE/global_fields.png]]

### High Stress
![[Runs/Fisso/u,v,p solo Data + only stress PDE/high_stress.png]]

### L2 Errors History
![[Runs/Fisso/u,v,p solo Data + only stress PDE/l2_errors_history.png]]

### Loss History
![[Runs/Fisso/u,v,p solo Data + only stress PDE/loss_history.png]]

## 📜 Log della Run
- [[Runs/Fisso/u,v,p solo Data + only stress PDE/train_log.txt]]

## 📌 Note e Conclusioni
- 
