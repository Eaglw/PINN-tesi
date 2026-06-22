---
date: 2026-06-21T14:33:33
inverse_problem: false
dataset: 4_roll_mill.csv
epochs: 100000
staged: true
Computer: Maurizio
---
# Idea
Rispetto a [[entrambe PDE + u,v data]] che comunque non stava convergendo ho spento la momentum, e fatto fare 100k epoche su dati + solo stress. 
# Run_001_4_roll_mill_L8x128_E100000_SiLU_stagedTrue_invFalse_20260620_221118

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
| **Data Loss** | 8.261107e-06 |
| **Boundary Loss** | 1.045694e+00 |
| **BC_u** | 2.140166e-06 |
| **BC_v** | 2.062624e-06 |
| **BC_p** | 1.045689e+00 |
| **Momentum Loss** | 4.723411e+02 |
| **Constitutive Loss** | 6.968186e-07 |
| **Total PDE Loss** | 4.723411e+02 |
| **Total Loss** | 1.419115e+03 |
| **Mean Abs f_u** | 1.789143e+01 |
| **Mean Abs f_v** | 1.801855e+01 |
| **Mean Abs f_txx** | 5.816143e-04 |
| **Mean Abs f_txy** | 6.068649e-04 |
| **Mean Abs f_tyy** | 5.580868e-04 |

### 🔹 Errori Relativi L2
| Variabile | Errore L2 |
|---|---|
| **u** | 0.002872 |
| **v** | 0.002876 |
| **p** | 1.000000 |
| **tau_xx** | 0.044341 |
| **tau_xy** | 0.027379 |
| **tau_yy** | 0.042992 |
| **tau_xx_masked** | 0.044340 |
| **tau_xy_masked** | 0.027373 |
| **tau_yy_masked** | 0.042991 |

## 🖼️ Plot Generati

### Global Fields
![[Runs/Maurizio/u,v data + only stress PDe/global_fields.png]]

### High Stress
![[Runs/Maurizio/u,v data + only stress PDe/high_stress.png]]

### L2 Errors History
![[Runs/Maurizio/u,v data + only stress PDe/l2_errors_history.png]]

### Loss History
![[Runs/Maurizio/u,v data + only stress PDe/loss_history.png]]

### Params Evolution
![[params_evolution.png]]

## 📜 Log della Run
- [[Runs/Maurizio/u,v data + only stress PDe/train_log.txt]]

## 📌 Note e Conclusioni
-  Mi sembra che stia funzionando particolarmente bene, in teoria sembra che con più epoche sarebbe potuta anche finire meglio, ma questo è il test più semplice. 
