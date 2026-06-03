# Diario Sperimentale: Debugging Viscoelastic PINN

Questo documento riassume tutte le prove, le ipotesi e le scoperte matematiche fatte per risolvere la degradazione dei parametri e la mancata convergenza del campo di stress.

## 1. Il Problema Iniziale (Il Collasso dei Parametri)
* **Sintomi:** Nel problema inverso (Goal 1), la rete imparava lo stress solo vicino all'inlet. All'interno del dominio prevedeva $\tau = 0$ e faceva crollare la viscosità polimerica ($\mu_p \to 0$).
* **Diagnosi:** La rete stava sfruttando una "scappatoia" matematica. Siccome le equazioni iperboliche dello stress sono difficili da integrare, la rete azzerava $\mu_p$ per far scomparire il termine sorgente, ottenendo una loss PDE perfettamente nulla ($0=0$) senza sforzo.

## 2. Test Goal 0 (Pure Physics - La Prova del Nove)
* **Setup:** Parametri fissati ai valori veri, Equazione del Momento e Costitutiva accese simultaneamente dall'epoca 0, nessuna Data Loss nel dominio.
* **Risultati:** La rete ha imparato **perfettamente** lo stress in sole 5.000 epoche.
* **Conclusione Fondamentale:** Architettura, adimensionalizzazione e dataset sono matematicamente **perfetti**. Il fallimento del Goal 1 è dovuto esclusivamente a come viene gestito il training inverso (Staged Training) e lo sblocco dei parametri. Nel Goal 1, il Momento è spento nella Fase 1, togliendo il vincolo globale della vorticità e permettendo alla rete di barare.

## 3. Test Precondizionamento (Divisione per $\beta_{poly}$)
* **Setup:** Abbiamo diviso l'equazione costitutiva per $\beta_{poly}$ per isolare il termine sorgente (es. da $-2\beta_{poly}u_x$ a $-2u_x$) e far esplodere la loss se $\mu_p \to 0$.
* **Risultati:** Disastro divergente. La viscosità polimerica $\mu_p$ è cresciuta a dismisura e i parametri `eps` e `alpha` non convergevano.
* **Conclusione:** Il trucco bloccava la scappatoia dello zero, ma creava un feedback positivo inverso. Se la rete sovrastimava lo stress iniziale, l'ottimizzatore ingrandiva $\mu_p$ per compensare la divisione, causando divergenza. (Modifica rimossa).

## 4. Test Warmup Lungo + Guess Maggiorato
* **Setup:** `GUESS_MULTIPLIER = 1.5` e parametri congelati molto più a lungo (Fase 1).
* **Risultati:** La rete ha imparato un $\tau_{xy}$ non nullo, ma con magnitudo sballata (oscillava tra 3 e -3 invece che 2 e -2).
* **Conclusione:** Questo ha confermato che **la rete è capace di imparare lo stress** se costretta dal congelamento. La magnitudo era sbagliata semplicemente perché la rete stava fittando il $\mu_p$ finto che le avevamo imposto. 

## 5. Test Goal 1 a Parametri Bloccati (Solo `eps` e `alpha` liberi)
* **Setup:** Parametri fisici corretti e congelati per sempre in `trainer.py`. Problema non-inverso per i reologici.
* **Risultati:** $\tau_{xy}$ converge bene. $\tau_{yy}$ impara un profilo spurio. $\tau_{xx}$ viene totalmente ignorato (piatto). Errori di velocità sui muri.
* **Diagnosi (Derivate Rumorose):** 
  * $\tau_{yy}$ spurio: Causato dagli errori di velocità ai muri ($v_y \neq 0$), che iniettano una sorgente falsa.
  * Perché $\tau_{xx}$ fallisce e $\tau_{xy}$ no: $\tau_{xy}$ dipende linearmente dalla derivata $u_y$. $\tau_{xx}$ dipende dal quadrato $(u_y)^2$. Forzare la rete a fittare i dati di velocità di COMSOL rende $u_y$ "rumoroso". Elevare al quadrato questo rumore crea un campo caotico che la rete si rifiuta di fittare, preferendo arresa ($\tau_{xx}=0$).

## 6. L'Intuizione della Geometria a Restrizione
* **Setup:** Geometria con contrazione (dove l'accelerazione $u_x$ non è nulla).
* **Risultati:** $\tau_{yy}$ assume valori ottimi. $\tau_{xy}$ continua a migliorare. $\tau_{xx}$ resta l'unico grosso problema.
* **Diagnosi Definitiva (Coil-Stretch Singularity):** 
  * Nella contrazione, $u_x$ e $v_y$ diventano le sorgenti lineari dirette per i componenti normali. Ecco perché $\tau_{yy}$ si stabilizza.
  * Tuttavia, per $\tau_{xx}$, il coefficiente diventa $(1 - 2Wi \cdot u_x)$. Nel fluido che accelera, questo termine si avvicina a zero (transizione coil-stretch). L'equazione PDE diventa iper-instabile e $\tau_{xx}$ esplode. Le PINN standard non riescono a risolvere questa singolarità fisica senza una griglia di collocazione eccezionalmente fitta attorno allo spigolo o l'ausilio dei dati esatti di stress.

---
> [!IMPORTANT]
> **Lo Scenario Attuale**
> 1. Abbiamo dimostrato matematicamente e numericamente che il codice, il dataset e le equazioni sono corretti (Goal 0 trionfa).
> 2. Abbiamo scoperto che risolvere il problema **semi-inverso** (trovare i parametri conoscendo solo la velocità) in flussi viscoelastici ad alto Weissenberg è al limite delle possibilità delle PINN standard, a causa del rumore sulle derivate e delle singolarità iperboliche.
> 3. L'unica strada garantita per estrarre in modo robusto e veloce i parametri in geometrie complesse è fornire alla rete l'intero dataset (`comsol_full`), permettendole di ancorarsi allo stress reale senza doverlo integrare dalle derivate caotiche della velocità.
