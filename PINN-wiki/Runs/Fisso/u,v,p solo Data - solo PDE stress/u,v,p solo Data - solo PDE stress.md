---
date: 2026-06-23T14:36:31
inverse_problem: false
dataset: 4_roll_mill.csv
epochs: "88000"
Computer: Fisso
staged: true
inverse: false
---

> [!NOTE] PROBLEMA
> Non ho impostato tra i parametri il lambda corretto, infatti era rimasto ad 1 invece che a 0.05, che è quello del dataset. 

# Run_001_4_roll_mill_StressOnly_L8x128_E1=50000_E2=30000_SiLU_invFalse_20260623_143606

## 💡 Idea
Dopo [[u,v,p soloData + entrambe PDE]] e alla stessa run con più epoche [[u,v,p solo data 50+5k - entrambe PDE 30+3k]], ho pensato che la concorrenza di momentum e costitutive fosse il problema, quindi ho spento la momentum per vedere se potesse convergere. 

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
| **Data Loss** | 2.083460e-06 |
| **Boundary Loss** | 9.347685e-05 |
| **BC_u** | 5.044869e-05 |
| **BC_v** | 3.591101e-05 |
| **BC_p** | 7.117146e-06 |
| **Momentum Loss** | 1.998142e+04 |
| **Constitutive Loss** | 3.412588e-04 |
| **Total PDE Loss** | 1.998142e+04 |
| **Total Loss** | 5.994426e+04 |
| **Mean Abs f_u** | 1.234161e+02 |
| **Mean Abs f_v** | 1.222238e+02 |
| **Mean Abs f_txx** | 1.424490e-02 |
| **Mean Abs f_txy** | 1.126642e-02 |
| **Mean Abs f_tyy** | 1.420167e-02 |

### 🔹 Errori Relativi L2
| Variabile | Errore L2 |
|---|---|
| **u** | 0.001439 |
| **v** | 0.001448 |
| **p** | 0.003066 |
| **tau_xx** | 4.817967 |
| **tau_xy** | 3.057825 |
| **tau_yy** | 4.824102 |
| **tau_xx_masked** | 4.817964 |
| **tau_xy_masked** | 3.057827 |
| **tau_yy_masked** | 4.824096 |

## 🖼️ Plot Generati

### Global Fields
![[Runs/Fisso/u,v,p solo Data - solo PDE stress/global_fields.png]]

### High Stress
![[Runs/Fisso/u,v,p solo Data - solo PDE stress/high_stress.png]]

### L2 Errors History
![[Runs/Fisso/u,v,p solo Data - solo PDE stress/l2_errors_history.png]]

### Loss History
![[Runs/Fisso/u,v,p solo Data - solo PDE stress/loss_history.png]]

## 📜 Log della Run
- [[Runs/Fisso/u,v,p solo Data - solo PDE stress/train_log.txt]]

## 📌 Note e Conclusioni
- I risultati fanno davvero pena, una loss che scende ma con un errore L2 sullo stress che addirittura sale. Come è possibile? 
- Trovato errore sul lambda, quindi magari vale la pena riprovarci con il lambda corretto, vediamo. 
