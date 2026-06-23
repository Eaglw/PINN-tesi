---
date: 2026-06-21T18:29:02
inverse_problem: false
dataset: 4_roll_mill.csv
epochs: 88000
Computer: Fisso
staged: false
inverse: false
---
# Idea
Dopo [[u,v,p soloData + entrambe PDE]] con poche epoche, 30k dati e 10k stress, ho provato ad allungare, con 50k adam + 5k L-BFGS per i dati di u,v,p e poi 30k adam e 3k L-BFGS per lo stress. Per lo stress erano attive nella loss sia la momentum che la costitutive. 
# Run_001_4_roll_mill_StressOnly_L8x128_E1=50000_E2=30000_SiLU_invFalse_20260621_125112

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
| **lam** | 1.000000 |
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
| **Momentum Loss** | 3.512666e-03 |
| **Constitutive Loss** | 1.701938e-02 |
| **Total PDE Loss** | 2.053204e-02 |
| **Total Loss** | 6.177750e-02 |
| **Mean Abs f_u** | 3.618743e-02 |
| **Mean Abs f_v** | 3.651462e-02 |
| **Mean Abs f_txx** | 9.450989e-02 |
| **Mean Abs f_txy** | 1.276964e-01 |
| **Mean Abs f_tyy** | 9.537764e-02 |

### 🔹 Errori Relativi L2
| Variabile | Errore L2 |
|---|---|
| **u** | 0.001211 |
| **v** | 0.001205 |
| **p** | 0.002821 |
| **tau_xx** | 1.064884 |
| **tau_xy** | 1.032975 |
| **tau_yy** | 1.079575 |
| **tau_xx_masked** | 1.064837 |
| **tau_xy_masked** | 1.032915 |
| **tau_yy_masked** | 1.079533 |

## 🖼️ Plot Generati

### Global Fields
![[Runs/Fisso/Run_001_4_roll_mill_StressOnly_L8x128_E1=50000_E2=30000_SiLU_invFalse_20260621_125112/global_fields.png]]

### High Stress
![[Runs/Fisso/Run_001_4_roll_mill_StressOnly_L8x128_E1=50000_E2=30000_SiLU_invFalse_20260621_125112/high_stress.png]]

### L2 Errors History
![[Runs/Fisso/Run_001_4_roll_mill_StressOnly_L8x128_E1=50000_E2=30000_SiLU_invFalse_20260621_125112/l2_errors_history.png]]

### Loss History
![[Runs/Fisso/Run_001_4_roll_mill_StressOnly_L8x128_E1=50000_E2=30000_SiLU_invFalse_20260621_125112/loss_history.png]]

## 📜 Log della Run
- [[Runs/Fisso/Run_001_4_roll_mill_StressOnly_L8x128_E1=50000_E2=30000_SiLU_invFalse_20260621_125112/train_log.txt]]

## 📌 Note e Conclusioni
- Credo proprio che addestrare sia momentum che constitutive insieme non sia una grande idea, penso che non ci si riesca a districare tra le due se sono attive contemporaneamente. 
# next run
[[u,v,p solo Data - solo PDE stress - wrong lambda]]