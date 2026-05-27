---
title: "Viscosity model based on Giesekus equation"
source: "https://www.degruyterbrill.com/document/doi/10.1515/arh-2024-0004/html"
author:
  - "[[Sun Kyoung Kim]]"
published: 2024-01-01
created: 2026-05-06
description: "This work presents a viscosity model based on the Giesekus equation. The model is shown to be more flexible than the Cross and Carreau models in representing the shear-thinning behavior of viscoelastic fluids. It has been investigated that the influence of the model parameters on the viscosity showed that the mobility parameter α plays a distinctive role in adjusting the inflection shape of the viscosity curve. The results show that the new model is able to accurately capture the shear-thinning behavior of polystyrene data, while the Cross and Carreau models tend to underestimate and overestimate the viscosity at the inflection point, respectively. It has been also shown that the Yasuda-type modification is also applicable to the proposed model. Moreover, the viscosity model has been applied to simultaneously fitting a polymeric liquid system and a particulate slurry system. The new viscosity model is a promising tool for modeling the shear-thinning behavior of viscoelastic fluids in a wide range of applications."
tags:
  - "clippings"
---
This work presents a viscosity model based on the Giesekus equation. The model is shown to be more flexible than the Cross and Carreau models in representing the shear-thinning behavior of viscoelastic fluids. It has been investigated that the influence of the model parameters on the viscosity showed that the mobility parameter *α* plays a distinctive role in adjusting the inflection shape of the viscosity curve. The results show that the new model is able to accurately capture the shear-thinning behavior of polystyrene data, while the Cross and Carreau models tend to underestimate and overestimate the viscosity at the inflection point, respectively. It has been also shown that the Yasuda-type modification is also applicable to the proposed model. Moreover, the viscosity model has been applied to simultaneously fitting a polymeric liquid system and a particulate slurry system. The new viscosity model is a promising tool for modeling the shear-thinning behavior of viscoelastic fluids in a wide range of applications.

Keywords: [viscosity model](https://www.degruyterbrill.com/search?query=keywordValues%3A%28%22viscosity%20model%22%29%20AND%20journalKey%3A%28%22ARH%22%29&documentVisibility=all&documentTypeFacet=article); [Giesekus equation](https://www.degruyterbrill.com/search?query=keywordValues%3A%28%22Giesekus%20equation%22%29%20AND%20journalKey%3A%28%22ARH%22%29&documentVisibility=all&documentTypeFacet=article); [shear thinning](https://www.degruyterbrill.com/search?query=keywordValues%3A%28%22shear%20thinning%22%29%20AND%20journalKey%3A%28%22ARH%22%29&documentVisibility=all&documentTypeFacet=article); [group viscosity](https://www.degruyterbrill.com/search?query=keywordValues%3A%28%22group%20viscosity%22%29%20AND%20journalKey%3A%28%22ARH%22%29&documentVisibility=all&documentTypeFacet=article)

Viscosity models were initially constructed based on theoretical foundations, but their practical utility primarily centers around their ability to effectively represent experimental data \[[^15],[^16]\]. The Cross model originated from the chain linkage model \[[^17],[^18]\], while the Carreau model was formulated by selecting a function that simplified expressions derived from molecular network theory \[[^19]\]. Subsequently, Yasuda introduced an index to enhance the characteristics of the Carreau model \[[^20],[^21]\]. In this study, a novel model is introduced, derived from a differential constitutive equation, aiming at developing a more versatile viscosity model. In general, the viscosity models find extensive utility in analytical \[[^22],[^23],[^24]\], numerical \[[^25],[^26]\], and empirical \[[^27]\] investigations across diverse applications. Notably, within the realm of polymeric liquids, the Cross and Carreau models serve as fundamental tools for modeling viscosities. Furthermore, in polymer processes such as 3D printing \[[^28]\], injection molding \[[^29],[^30]\], and extrusion \[[^31]\], they have become the established standard for characterizing melt viscosities.

A shear viscosity can be determined by the ratio of shear stress to shear rate when applying a constitutive equation to simple shear flow \[[^15],[^16]\]. However, using a simplistic form of a constitutive equation may result in constant viscosity or unrealistic shear-thinning behavior, as observed in the Zaremba–Fromm–Dewitt (ZFD) model \[[^32]\]. Carreau derived his model from an integral constitutive equation \[[^33]\], with the index chosen to converge to a power-law relationship. However, it is sometimes challenging to fit an experimental data to the Carreau model \[[^34],[^35]\]. In this work, a new viscosity model is formulated by incorporating an index into the viscosity calculation derived from the Giesekus model \[[^36]\], resembling the relationship between the ZFD model and the Carreau model. The attributes of the developed model are then tested, evaluated, and discussed in terms of its practical applicability.

The power-law viscosity with consistency *K* and index *n* can be written again as

(1)

where the zero-shear viscosity is given by and is the characteristic time. For *n* = 1, a Newtonian viscosity is resulted. When , shear thinning occurs, whereas shear thickening does for . As *n* approaches 0, shear thinning becomes severer. Let us introduce a dimensionless shear rate of the form:

(2)

Using equations ([^2]) and ([^1]) can be written again as

(3)

where can be treated as the Weissenberg number when is a physical relaxation time for elasticity. In the viscosity models, is occasionally chosen to simply non-dimensionalize the shear rate. It should also be acknowledged that the power-law model remains commonly utilized for representing viscosity data in practical applications despite the existence of more complex models \[[^37]\].

Based on the molecular network theory, Carreau derived the viscosity of a steady simple shear flow as . Furthermore, by defining *f* <sub><em>p</em> </sub> and *g* <sub><em>p</em> </sub> as functions of independent shear rates *f* and *g*, the viscosity can be calculated as . When *f* and *g* are appropriately chosen to account for other rheological conditions, the viscosity can be described by . When , it asymptotically approaches equation ([^1]), leading to the Carreau viscosity model \[[^19]\], which is

(4)

where

(5)

where is the modified dimensionless shear rate in the Carreau model. It should be emphasized that *λ* designates the onset point at which the nonlinear viscous behavior starts. Consider that a fluid follows the ZFD model, which follows \[[^15],[^36]\]

(6)

where , , and are the stress tensor, strain rate tensor, and the Jaumann derivative, respectively. For a simple shear flow, the viscosity is obtained as \[[^38]\]

(7)

Using the aforementioned expression, equation ([^3]) can be written again as

(8)

This is the way the Carreau viscosity model is related to the ZFD equation.

Now, let us describe the Giesekus viscoelastic model \[[^36]\], which is one of the popular nonlinear models \[[^39]\]. Using the upper-convected derivative , the model is of the form

(9)

where , , and is the mobility parameter. Especially in polymer melt, this parameter is related to chain response to the flow. It quantifies the capability of chain being stretched along the flow. With *α* = 0, this is reduced to the upper-convected Maxwell model. Also, it approximates the ZFD model for *α* = 1. Refer to the study by Wiest and Bird \[[^40]\] for the interpretation and extension of the Giesekus model.

For a simple shear flow, Giesekus obtained the viscosity of the form \[[^36],[^38]\]

(10)

where

(11)

and

(12)

When *α* is set to ½, signifying a state of neutral mobility, we observe the result as

(13)

Interestingly, a continued fraction derived from equation ([^5]) denoted as

(14)

is identical to , which can be obtained by solving a quadratic equation.

Based on the analogy in equation ([^6]), this work proposes another viscosity model that asymptotes to the power-law model:

(15)

Introducing a truncation term , the viscosity becomes a five-parameter model \[[^41]\]:

(16)

The characteristics of the proposed model have been investigated as the parameters are changed. It is observed to exhibit a wider span depending on *n*, similar to the characteristics seen in other existing models, as shown in [Figure 1](#j_arh-2024-0004_fig_001). Furthermore, the emergence of parallel lines shown in [Figure 2](#j_arh-2024-0004_fig_002) depending on the *λ* is a well-established phenomenon. It should be emphasized that *λ* is not merely a parameter but one endowed with physical significance, as it is specifically designated as the longest relaxation time in the derivation of the Carreau model.

![Figure 1 
                  Variation of viscosity alongside n by equation (15) with α = 0.5.
               ](https://www.degruyterbrill.com/document/doi/10.1515/arh-2024-0004/asset/graphic/j_arh-2024-0004_fig_001.jpg)

Figure 1

Variation of viscosity alongside *n* by equation ([^9]) with *α* = 0.5.

![Figure 2 
                  Variation of viscosity alongside λ with α = 0.5 and n = 0.5.
               ](https://www.degruyterbrill.com/document/doi/10.1515/arh-2024-0004/asset/graphic/j_arh-2024-0004_fig_002.jpg)

Figure 2

Variation of viscosity alongside *λ* with *α* = 0.5 and *n* = 0.5.

However, the parameter *α* represents a distinctive feature unique to this model, allowing for a finer adjustment of the inflection shape occurring in the region where shear thinning occurs. Furthermore, this underscores its origin from the mobility of the Giesekus equation.

As illustrated in [Figure 3](#j_arh-2024-0004_fig_003), this model exhibits intermediate characteristics between the Carreau model and the Cross model. The Carreau model tends to represent the region of steep shear-thinning more effectively than the Cross model, while the proposed model aligns more closely with the Carreau model. When fitting viscosity curves from experimental data, it is desirable to first determine and *n* separately for the low shear rate and high shear rate regions before determining *λ* through fitting. However, in many cases, due to experimental limitations, and *n* may need to be fitted simultaneously without being pre-determined.

![Figure 3 
                  Comparison of power-law, equation (15) with α = 0.5, Cross and Carreau models for n = 0.3.
               ](https://www.degruyterbrill.com/document/doi/10.1515/arh-2024-0004/asset/graphic/j_arh-2024-0004_fig_003.jpg)

Figure 3

Comparison of power-law, equation ([^9]) with *α* = 0.5, Cross and Carreau models for *n* = 0.3.

In cases where the dataset is insufficient in size, the guarantee of uniqueness for the coefficients cannot be ensured. Indeed, when all parameters are simultaneously determined, results that appear to better match the experimental data are typically obtained. However, this alignment does not necessarily conform to the original intent of models designed to obtain at low shear rates and to achieve the asymptote of the power-law model at high shear rates. Instead, it merely serves as a mathematical tool for the convenient utilization of viscosity values. Of course, in situations where viscosity function is essential, such as in numerical analyses, it might be inevitably required. Nevertheless, when applied beyond the range within which experimental data have been obtained, it exposes itself to the risk of extrapolation.

In [Figure 4](#j_arh-2024-0004_fig_004), viscosity values for various models under simple shear flow conditions are presented. The viscosity values obtained from the ZFD model exhibit unrealistic shear-thinning behavior, rendering them unsuitable for immediate application. Notably, the viscosity values derived from the Giesekus model closely resemble those of the Carreau model with *n* = 0. In this study, the modified dimensionless shear rates proposed in equations ([^4]) and ([^8]) were deemed to be equivalent, as confirmed by the results presented here. Also, note that certain materials show similar behaviors \[[^42]\]. For the case of *n* = 0.5 and *α* = 0.5, resembling the Carreau model, and *α* = 0.9, indicating higher mobility, we observe differences in the viscosity curves, with the latter exhibiting a more pronounced shear thinning. In addition, it can observed that the Cross model deviates significantly from other models. This disparity arises in that the proposed model holds a characteristic distinct from the Cross model and akin to the Carreau model.

![Figure 4 
                  Comparison of various viscosities during simple shear flow.
               ](https://www.degruyterbrill.com/document/doi/10.1515/arh-2024-0004/asset/graphic/j_arh-2024-0004_fig_004.jpg)

Figure 4

Comparison of various viscosities during simple shear flow.

These trends are also evident in the results of fitting experimental polystyrene data presented in [Figure 5](#j_arh-2024-0004_fig_005). Fitting results share coefficients for the Cross, Carreau, and equation ([^9]) models show distinctive behaviors. Specifically, the Cross model tends to underestimate at the inflection point, while the Carreau model tends to overestimate, whereas results based on equation ([^9]) pass through an intermediate point between the Carreau model and the experimental data. When viscosity is fitted using equation ([^9]), it exhibits a characteristic that influences the inflection point itself. Beyond an *α* -value of 0.5, as observed in [Figure 6](#j_arh-2024-0004_fig_006), the lines become parallel to each other, mirroring the trend observed in [Figure 2](#j_arh-2024-0004_fig_002). In case a characteristic similar to the *λ* in the Carreau model is strictly required, it is recommended to adjust the *λ* using the following equation:

(17)

where the coefficients have been obtained by fitting equations ([^9]) to ([^3]).

![Figure 5 
                  Fitting viscosity of linear polystyrene (molecular weight 2,000,000) solution in 1-chloronaphtalene measured using a capillary rheometer at 25°C by equation (15) with α = 0.5, Cross and Carreau model [6].
               ](https://www.degruyterbrill.com/document/doi/10.1515/arh-2024-0004/asset/graphic/j_arh-2024-0004_fig_005.jpg)

Figure 5

Fitting viscosity of linear polystyrene (molecular weight 2,000,000) solution in 1-chloronaphtalene measured using a capillary rheometer at 25°C by equation ([^9]) with *α* = 0.5, Cross and Carreau model \[[^20]\].

![Figure 6 
                  Variation of viscosity alongside α  with n = 0.5.
               ](https://www.degruyterbrill.com/document/doi/10.1515/arh-2024-0004/asset/graphic/j_arh-2024-0004_fig_006.jpg)

Figure 6

Variation of viscosity alongside *α* with *n* = 0.5.

When is used applying the aforementioned correction, the results presented in [Figure 7](#j_arh-2024-0004_fig_007) are obtained. The viscosity model described by equation ([^9]) converges to the power-law model while also simulating cases where shear thinning occurs more steeply than the power-law at the inflection point. As evident in [Figure 7](#j_arh-2024-0004_fig_007), when alpha *α* exceeds 0.5, it converges toward the power-law model from the outside of the power-law model. In situations where there is a region where the rate of shear thinning exceeds that of the power-law model, modeling viscosity using this model would be advantageous.

 $Figure 7 
                  Adjusting the viscosity curve with 
                        
                           
                           
                              λ
                              ′
                           
                           \lambda ^{\prime} 
                        
                      alongside α.$ 

Figure 7

Adjusting the viscosity curve with alongside *α*.

Yasuda attempted to address the discrepancy depicted in [Figure 5](#j_arh-2024-0004_fig_005) by introducing an additional index denoted as *a* \[[^21]\]. Let us now investigate the determination of Γ in the Cross model. While a lot of elaborations are made to build accurate viscoelastic constitutive equations, some simply modified the power-law model in an empirical way. Dunleavy and Middleman proposed to modify to \[[^43]\], and later, Brewster and Irvine reused the same model in a form of and called it the modified power-law model \[[^44]\]. Note that such an expression is still employed in recent works \[[^45]\]. Of course, this is equivalent to the Cross model without the truncation term. Cross derived a viscosity function as from the chain linkage model relating *a* to \[[^18]\]. Here, considering the order to , it can be modified to

(18)

For *a* = 2, is attained. Substituting equation ([^12]) into equation ([^3]), we have the Carreau–Yasuda model expressed as

(19)

Let us incorporate the Yasuda-type index to equation ([^7]):

(20)

Analogously to equation ([^13]), equation ([^10]) can undergo additional modifications to transform into a six-parameter model structured as

(21)

where is the infinite-shear viscosity. This work has explained it by equation ([^12]) but the index *a* was initially employed to simply compensate error present in the Carreau model near the critical shear rate \[[^21]\]. As Yasuda suggested, the model presented in equation ([^9]) can also be used to control the curvature at the inflection point by adjusting parameter *a,* as shown in [Figure 8](#j_arh-2024-0004_fig_008). As *a* increases, the model well follows the truncated power-law model. At *a* = 8, the model almost realizes the sharp edge at Γ = 1. However, it should be mentioned that finding a clear physical interpretation for this parameter beyond its utility in fitting experimental data remains challenging. Therefore, it is considered more appropriate to vary the mobility parameter alpha while keeping *a* at a value of 2.

![Figure 8 
                  Viscosity by equation (15) with α = 0.5 alongside a.
               ](https://www.degruyterbrill.com/document/doi/10.1515/arh-2024-0004/asset/graphic/j_arh-2024-0004_fig_008.jpg)

Figure 8

Viscosity by equation ([^9]) with *α* = 0.5 alongside *a*.

Sometimes it is useful to simultaneously fit viscosities of a set. For instance, consider measuring the viscosities of filled polypropylene (PP) melts with varying loadings of fillers, as shown in [Figure 9](#j_arh-2024-0004_fig_009) \[[^46]\]. It is found that viscosity curves are parallel with each other at high shear rates. Thus, the same *n* value can be shared for multi-walled carbon nanotube (MWCNT)/PP composites with different loadings. Here, considering *λ* as the characteristic time shared by this material system, the viscosity for each filler loading can be expressed depending on *η* <sub><em>0</em> </sub> and *α*. The obtained constants are shown in [Figure 9](#j_arh-2024-0004_fig_009). The viscosity curves obtained match well with experimental data, as can be seen in the figure. The values of *α* increase along with the MWCNT loading as the increased loading results in higher mobility under the given *η* <sub><em>0</em> </sub> .

 $Figure 9 
                  Viscosity of MWCNT-filled PP at different loading levels (wt%) with variations in α and 
                        
                           
                           
                              
                                 
                                    η
                                 
                                 
                                    0
                                 
                              
                           
                           {\eta }_{0}
                        
                      sharing n and λ [32]. The viscosity is measured at 200°C using a capillary rheometer.$ 

Figure 9

Viscosity of MWCNT-filled PP at different loading levels (wt%) with variations in *α* and sharing *n* and *λ* \[[^46]\]. The viscosity is measured at 200°C using a capillary rheometer.

In complex fluid systems, coefficients sharing as in [Figure 9](#j_arh-2024-0004_fig_009) are infeasible. Let us investigate such a case where all the viscosity coefficients have to be evaluated individually. An anode slurry system investigated by Bitsch et al. is modeled by the proposed viscosity model here \[[^47]\]. As shown in [Figure 10](#j_arh-2024-0004_fig_010), the viscosity model follows the experimental data well. Alongside, increase of the solid volume fraction, and *λ* increases and *n* decreases consistently. In the meantime, the value of *α* is smallest at 20% fraction for both with and without octanol addition. It is found to be useful in rheologically characterizing a fluid system.

![Figure 10 
                  Fitted viscosity of anode slurry with different volume % of solid particles with and without octanol addition at room temperature [33].
               ](https://www.degruyterbrill.com/document/doi/10.1515/arh-2024-0004/asset/graphic/j_arh-2024-0004_fig_010.jpg)

Figure 10

Fitted viscosity of anode slurry with different volume % of solid particles with and without octanol addition at room temperature \[[^47]\].

In the previous sections, equation ([^14]) has been suggested, which comprises six parameters. [Table 1](#j_arh-2024-0004_tab_001) presents the denotation and role of each parameter. Based on the investigations, this work recommends to initially fit the data using the model with *α* = 0.5. If controlling the onset of shear thinning by adjusting is enough, can simply remain 0.5. If required, change *α* according to the experimental data. If the correction for *λ* is correspondingly required, adjust the value using equation ([^11]). Still the discrepancy is observed introduce the Yasuda parameter *a* to adjust the behavior near the inflection point. As summarized in [Table 1](#j_arh-2024-0004_tab_001), *α* smoothly shifts the viscosity curve at high shear rates in parallel, while *a* alters the transition curvature from zero-shear viscosity to power-law viscosity. The onset of shear thinning would greatly vary along with the materials under test. Therefore, the described procedure is necessary to deal with various viscosity curves.

Table 1

Role of each parameter in equation ([^14])

|  | Denotation | Role |
| --- | --- | --- |
|  | Zero-shear viscosity | Viscosity at the low shear rate |
|  | Infinite-shear viscosity | Truncation at the high shear rate |
|  | Index | Index of the power-law model approaching asymptotically at high shear ([Figure 1](#j_arh-2024-0004_fig_001)) |
|  | Characteristic time | Control of starting point of nonlinear behavior ([Figure 2](#j_arh-2024-0004_fig_002)) |
|  | Mobility parameter | Parallel displacement from the power-law model at high shear ([Figure 6](#j_arh-2024-0004_fig_006)) |
|  | YASUDA parameter | Curvature of nonlinear transition ([Figure 8](#j_arh-2024-0004_fig_008)) |

This study has presented a new viscosity model based on the Giesekus equation. The authors investigated the influence of the model parameters on the viscosity and showed that the parameter *α* plays a distinctive role in adjusting the inflection shape of the viscosity curve. The results showed that the new model is able to accurately capture the shear-thinning behavior of polystyrene data, while the Cross and Carreau models tend to underestimate and overestimate the viscosity at the inflection point, respectively. In addition, the proposed model could be applied to fitting groups of viscosities. The author concludes that the new viscosity model is a promising tool for modeling the shear thinning behavior of viscoelastic fluids in a wide range of applications.

1. **Funding information:** This work was supported by NRF grants funded from the Korea government (NRF-2018R1A5A1024127 and 2020R1I1A2065650).
2. **Author contributions:** The author confirms the sole responsibility for the conception of the study, presented results and manuscript preparation.
3. **Conflict of interest:** The authors declare no conflict of interest.
4. **Ethical approval:** The conducted research is not related to either human or animal use.
5. **Data availability statement:** All data generated or analyzed during this study are included in this published article.

\[1\] Carreau PJ, De Kee DC, Chhabra RP. Rheology of polymeric systems: principles and applications. München: Carl Hanser; 2021.[10.3139/9781569907238.fm](https://doi.org/10.3139/9781569907238.fm) [Search in Google Scholar](https://scholar.google.com/scholar?q=Carreau%20PJ%2C%20De%20Kee%20DC%2C%20Chhabra%20RP.%20Rheology%20of%20polymeric%20systems%3A%20principles%20and%20applications.%20M%C3%BCnchen%3A%20Carl%20Hanser%3B%202021.)

\[2\] Osswald T, Rudolph N. Polymer rheology. München: Carl Hanser; 2015.[10.1007/978-1-56990-523-4](https://doi.org/10.1007/978-1-56990-523-4) [Search in Google Scholar](https://scholar.google.com/scholar?q=Osswald%20T%2C%20Rudolph%20N.%20Polymer%20rheology.%20M%C3%BCnchen%3A%20Carl%20Hanser%3B%202015.)

\[3\] Cross MM. Rheology of non-newtonian fluids: a new flow equation for pseudoplastic systems. J Colloid Sci. 1965;20(5):417–37. [10.1016/0095-8522(65)90022-X](https://doi.org/10.1016/0095-8522\(65\)90022-X).[Search in Google Scholar](https://scholar.google.com/scholar?q=Cross%20MM.%20Rheology%20of%20non-newtonian%20fluids%3A%20a%20new%20flow%20equation%20for%20pseudoplastic%20systems.%20J%20Colloid%20Sci.%201965%3B20%285%29%3A417%E2%80%9337.%2010.1016%2F0095-8522%2865%2990022-X%20.)

\[4\] Cross MM. Polymer rheology: influence of molecular weight and polydispersity. J Appl Polym Sci. 1969;13(4):765–74. [10.1002/app.1969.070130415](https://doi.org/10.1002/app.1969.070130415).[Search in Google Scholar](https://scholar.google.com/scholar?q=Cross%20MM.%20Polymer%20rheology%3A%20influence%20of%20molecular%20weight%20and%20polydispersity.%20J%20Appl%20Polym%20Sci.%201969%3B13%284%29%3A765%E2%80%9374.%2010.1002%2Fapp.1969.070130415%20.)

\[5\] Carreau PJ. Rheological equations from molecular network theories. Trans Soc Rheology. 1972;16(1):99–127. [10.1122/1.549276](https://doi.org/10.1122/1.549276).[Search in Google Scholar](https://scholar.google.com/scholar?q=Carreau%20PJ.%20Rheological%20equations%20from%20molecular%20network%20theories.%20Trans%20Soc%20Rheology.%201972%3B16%281%29%3A99%E2%80%93127.%2010.1122%2F1.549276%20.)

\[6\] Yasuda K. Investigation of the analogies between viscometric and linear viscoelastic properties of polystyrene fluids \[dissertation\]. Cambridge (MA): Massachusetts Institute of Technology; 1979. https://dspace.mit.edu/handle/1721.1/16043.[Search in Google Scholar](https://scholar.google.com/scholar?q=Yasuda%20K.%20Investigation%20of%20the%20analogies%20between%20viscometric%20and%20linear%20viscoelastic%20properties%20of%20polystyrene%20fluids%20%5Bdissertation%5D.%20Cambridge%20%28MA%29%3A%20Massachusetts%20Institute%20of%20Technology%3B%201979.%20https%3A%2F%2Fdspace.mit.edu%2Fhandle%2F1721.1%2F16043.)

\[7\] Yasuda KY, Armstrong RC, Cohen RE. Shear flow properties of concentrated solutions of linear and star branched polystyrenes. Rheol Acta. 1981;20(2):163–78. [10.1007/BF01513059](https://doi.org/10.1007/BF01513059).[Search in Google Scholar](https://scholar.google.com/scholar?q=Yasuda%20KY%2C%20Armstrong%20RC%2C%20Cohen%20RE.%20Shear%20flow%20properties%20of%20concentrated%20solutions%20of%20linear%20and%20star%20branched%20polystyrenes.%20Rheol%20Acta.%201981%3B20%282%29%3A163%E2%80%9378.%2010.1007%2FBF01513059%20.)

\[8\] Kim SK. Flow-rate based method for velocity of fully developed laminar flow in tubes. J Rheol. 2018 Nov 1;62(6):1397–407. [10.1122/1.5041958](https://doi.org/10.1122/1.5041958).[Search in Google Scholar](https://scholar.google.com/scholar?q=Kim%20SK.%20Flow-rate%20based%20method%20for%20velocity%20of%20fully%20developed%20laminar%20flow%20in%20tubes.%20J%20Rheol.%202018%20Nov%201%3B62%286%29%3A1397%E2%80%93407.%2010.1122%2F1.5041958%20.)

\[9\] Kim SK. Flow rate based framework for solving viscoplastic flow with slip. J Non-Newtonian Fluid Mech. 2019;269:37–46. [10.1016/j.jnnfm.2019.06.002](https://doi.org/10.1016/j.jnnfm.2019.06.002).[Search in Google Scholar](https://scholar.google.com/scholar?q=Kim%20SK.%20Flow%20rate%20based%20framework%20for%20solving%20viscoplastic%20flow%20with%20slip.%20J%20Non-Newtonian%20Fluid%20Mech.%202019%3B269%3A37%E2%80%9346.%2010.1016%2Fj.jnnfm.2019.06.002%20.)

\[10\] Kim SK, Kazmer DO, Colon AR, Coogan TJ, Peterson AM. Non-Newtonian modeling of contact pressure in fused filament fabrication. J Rheol. 2021;65(1):27–42. [10.1122/8.0000052](https://doi.org/10.1122/8.0000052).[Search in Google Scholar](https://scholar.google.com/scholar?q=Kim%20SK%2C%20Kazmer%20DO%2C%20Colon%20AR%2C%20Coogan%20TJ%2C%20Peterson%20AM.%20Non-Newtonian%20modeling%20of%20contact%20pressure%20in%20fused%20filament%20fabrication.%20J%20Rheol.%202021%3B65%281%29%3A27%E2%80%9342.%2010.1122%2F8.0000052%20.)

\[11\] Hong J, Kim SK, Cho YH. Flow and solidification of semi-crystalline polymer during micro-injection molding. Int J Heat Mass Transf. 2020;153:119576. [10.1016/j.ijheatmasstransfer.2020.119576](https://doi.org/10.1016/j.ijheatmasstransfer.2020.119576) [Search in Google Scholar](https://scholar.google.com/scholar?q=Hong%20J%2C%20Kim%20SK%2C%20Cho%20YH.%20Flow%20and%20solidification%20of%20semi-crystalline%20polymer%20during%20micro-injection%20molding.%20Int%20J%20Heat%20Mass%20Transf.%202020%3B153%3A119576.%2010.1016%2Fj.ijheatmasstransfer.2020.119576)

\[12\] Kim SK, Kazmer DO. Non-isothermal non-Newtonian three-dimensional flow simulation of fused filament fabrication. Addit Manuf. 2022;55:102833. [10.1016/j.addma.2022.102833](https://doi.org/10.1016/j.addma.2022.102833).[Search in Google Scholar](https://scholar.google.com/scholar?q=Kim%20SK%2C%20Kazmer%20DO.%20Non-isothermal%20non-Newtonian%20three-dimensional%20flow%20simulation%20of%20fused%20filament%20fabrication.%20Addit%20Manuf.%202022%3B55%3A102833.%2010.1016%2Fj.addma.2022.102833%20.)

\[13\] Kim SK. Collective viscosity model for shear thinning polymeric materials. Rheol Acta. 2020;59(1):63–72. [10.1007/s00397-019-01180-w](https://doi.org/10.1007/s00397-019-01180-w).[Search in Google Scholar](https://scholar.google.com/scholar?q=Kim%20SK.%20Collective%20viscosity%20model%20for%20shear%20thinning%20polymeric%20materials.%20Rheol%20Acta.%202020%3B59%281%29%3A63%E2%80%9372.%2010.1007%2Fs00397-019-01180-w%20.)

\[14\] Kazmer DO, Colon AR, Peterson AM, Kim SK. Concurrent characterization of compressibility and viscosity in extrusion-based additive manufacturing of acrylonitrile butadiene styrene with fault diagnoses. Addit Manuf. 2021;46:102106. [10.1016/j.addma.2021.102106](https://doi.org/10.1016/j.addma.2021.102106).[Search in Google Scholar](https://scholar.google.com/scholar?q=Kazmer%20DO%2C%20Colon%20AR%2C%20Peterson%20AM%2C%20Kim%20SK.%20Concurrent%20characterization%20of%20compressibility%20and%20viscosity%20in%20extrusion-based%20additive%20manufacturing%20of%20acrylonitrile%20butadiene%20styrene%20with%20fault%20diagnoses.%20Addit%20Manuf.%202021%3B46%3A102106.%2010.1016%2Fj.addma.2021.102106%20.)

\[15\] Kim SK, Jeong A. Numerical simulation of crystal growth in injection molded thermoplastics based on Monte Carlo method with shear rate tracking. Int J Precis Eng Manuf. 2019;20:641–50. [10.1007/s12541-019-00089-x](https://doi.org/10.1007/s12541-019-00089-x).[Search in Google Scholar](https://scholar.google.com/scholar?q=Kim%20SK%2C%20Jeong%20A.%20Numerical%20simulation%20of%20crystal%20growth%20in%20injection%20molded%20thermoplastics%20based%20on%20Monte%20Carlo%20method%20with%20shear%20rate%20tracking.%20Int%20J%20Precis%20Eng%20Manuf.%202019%3B20%3A641%E2%80%9350.%2010.1007%2Fs12541-019-00089-x%20.)

\[16\] Jung JS, Kim SK. Rapid numerical estimation of pressure drop in hot runner system. Micromachines. 2021;12(2):207. [10.3390/mi12020207](https://doi.org/10.3390/mi12020207).[Search in Google Scholar](https://scholar.google.com/scholar?q=Jung%20JS%2C%20Kim%20SK.%20Rapid%20numerical%20estimation%20of%20pressure%20drop%20in%20hot%20runner%20system.%20Micromachines.%202021%3B12%282%29%3A207.%2010.3390%2Fmi12020207%20.) [PubMed](https://pubmed.ncbi.nlm.nih.gov/33670694/) [PubMed Central](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7922069/)

\[17\] Tadmor Z, Gogos CG. Principles of polymer processing. New York: John Wiley & Sons; 2013.[Search in Google Scholar](https://scholar.google.com/scholar?q=Tadmor%20Z%2C%20Gogos%20CG.%20Principles%20of%20polymer%20processing.%20New%20York%3A%20John%20Wiley%20%26%20Sons%3B%202013.)

\[18\] DeWitt TW. A rheological equation of state which predicts non‐newtonian viscosity, normal stresses, and dynamic moduli. J Appl Phys. 1955;26(7):889–94. [10.1063/1.1722114](https://doi.org/10.1063/1.1722114).[Search in Google Scholar](https://scholar.google.com/scholar?q=DeWitt%20TW.%20A%20rheological%20equation%20of%20state%20which%20predicts%20non%E2%80%90newtonian%20viscosity%2C%20normal%20stresses%2C%20and%20dynamic%20moduli.%20J%20Appl%20Phys.%201955%3B26%287%29%3A889%E2%80%9394.%2010.1063%2F1.1722114%20.)

\[19\] Bird RB, Carreau PJ. A nonlinear viscoelastic model for polymer solutions and melts – I. Chem Eng Sci. 1968;23(5):427–34. [10.1016/0009-2509(68)87018-6](https://doi.org/10.1016/0009-2509\(68\)87018-6).[Search in Google Scholar](https://scholar.google.com/scholar?q=Bird%20RB%2C%20Carreau%20PJ.%20A%20nonlinear%20viscoelastic%20model%20for%20polymer%20solutions%20and%20melts%20%E2%80%93%20I.%20Chem%20Eng%20Sci.%201968%3B23%285%29%3A427%E2%80%9334.%2010.1016%2F0009-2509%2868%2987018-6%20.)

\[20\] Cho YI, Kensey KR. Effects of the non-newtonian viscosity of blood on flows in a diseased arterial vessel. Part 1: Steady flows. Biorheology. 1991;28(3-4):241–62. [10.3233/BIR-1991-283-415](https://doi.org/10.3233/BIR-1991-283-415).[Search in Google Scholar](https://scholar.google.com/scholar?q=Cho%20YI%2C%20Kensey%20KR.%20Effects%20of%20the%20non-newtonian%20viscosity%20of%20blood%20on%20flows%20in%20a%20diseased%20arterial%20vessel.%20Part%201%3A%20Steady%20flows.%20Biorheology.%201991%3B28%283-4%29%3A241%E2%80%9362.%2010.3233%2FBIR-1991-283-415%20.)

\[21\] Kelly NS, Gill HS, Cookson AN, Fraser KH. Influence of shear-thinning blood rheology on the laminar-turbulent transition over a backward facing step. Fluids. 2020;5(2):57. [10.3390/fluids5020057](https://doi.org/10.3390/fluids5020057).[Search in Google Scholar](https://scholar.google.com/scholar?q=Kelly%20NS%2C%20Gill%20HS%2C%20Cookson%20AN%2C%20Fraser%20KH.%20Influence%20of%20shear-thinning%20blood%20rheology%20on%20the%20laminar-turbulent%20transition%20over%20a%20backward%20facing%20step.%20Fluids.%202020%3B5%282%29%3A57.%2010.3390%2Ffluids5020057%20.)

\[22\] Giesekus H. A simple constitutive equation for polymer fluids based on the concept of deformation-dependent tensorial mobility. J Non-Newtonian Fluid Mech. 1982;11(1–2):69–109. [10.1016/0377-0257(82)85016-7](https://doi.org/10.1016/0377-0257\(82\)85016-7).[Search in Google Scholar](https://scholar.google.com/scholar?q=Giesekus%20H.%20A%20simple%20constitutive%20equation%20for%20polymer%20fluids%20based%20on%20the%20concept%20of%20deformation-dependent%20tensorial%20mobility.%20J%20Non-Newtonian%20Fluid%20Mech.%201982%3B11%281%E2%80%932%29%3A69%E2%80%93109.%2010.1016%2F0377-0257%2882%2985016-7%20.)

\[23\] Schaible T, Bonten C. In-line measurement and modeling of temperature, pressure, and blowing agent dependent viscosity of polymer melts. Appl Rheol. 2022;32(1):69–82. [10.1515/arh-2022-0123](https://doi.org/10.1515/arh-2022-0123).[Search in Google Scholar](https://scholar.google.com/scholar?q=Schaible%20T%2C%20Bonten%20C.%20In-line%20measurement%20and%20modeling%20of%20temperature%2C%20pressure%2C%20and%20blowing%20agent%20dependent%20viscosity%20of%20polymer%20melts.%20Appl%20Rheol.%202022%3B32%281%29%3A69%E2%80%9382.%2010.1515%2Farh-2022-0123%20.)

\[24\] Han CD. Rheology and processing of polymeric materials. Polymer rheology. Vol. 1, New York: Oxford University Press; 2007.[10.1093/oso/9780195187823.001.0001](https://doi.org/10.1093/oso/9780195187823.001.0001) [Search in Google Scholar](https://scholar.google.com/scholar?q=Han%20CD.%20Rheology%20and%20processing%20of%20polymeric%20materials.%20Polymer%20rheology.%20Vol.%201%2C%20New%20York%3A%20Oxford%20University%20Press%3B%202007.)

\[25\] Bird RB, Wiest JM. Constitutive equations for polymeric liquids. Annu Rev fluid Mech. 1995;27(1):169–93. [10.1146/annurev.fl.27.010195.001125](https://doi.org/10.1146/annurev.fl.27.010195.001125).[Search in Google Scholar](https://scholar.google.com/scholar?q=Bird%20RB%2C%20Wiest%20JM.%20Constitutive%20equations%20for%20polymeric%20liquids.%20Annu%20Rev%20fluid%20Mech.%201995%3B27%281%29%3A169%E2%80%9393.%2010.1146%2Fannurev.fl.27.010195.001125%20.)

\[26\] Wiest JM, Bird RB. Molecular extension from the Giesekus model. J non-Newtonian fluid Mech. 1986;22(1):115–9. [10.1016/0377-0257(86)80007-6](https://doi.org/10.1016/0377-0257\(86\)80007-6).[Search in Google Scholar](https://scholar.google.com/scholar?q=Wiest%20JM%2C%20Bird%20RB.%20Molecular%20extension%20from%20the%20Giesekus%20model.%20J%20non-Newtonian%20fluid%20Mech.%201986%3B22%281%29%3A115%E2%80%939.%2010.1016%2F0377-0257%2886%2980007-6%20.)

\[27\] Carreau PJ, Kee DD, Daroux M. An analysis of the viscous behaviour of polymeric solutions. Can J Chem Eng. 1979;57(2):135–40. [10.1002/cjce.5450570202](https://doi.org/10.1002/cjce.5450570202).[Search in Google Scholar](https://scholar.google.com/scholar?q=Carreau%20PJ%2C%20Kee%20DD%2C%20Daroux%20M.%20An%20analysis%20of%20the%20viscous%20behaviour%20of%20polymeric%20solutions.%20Can%20J%20Chem%20Eng.%201979%3B57%282%29%3A135%E2%80%9340.%2010.1002%2Fcjce.5450570202%20.)

\[28\] Ellwanger F, Georgantopoulos CK, Karbstein HP, Wilhelm M, Azad Emin M. Application of the ramp test from a closed cavity rheometer to obtain the steady-state shear viscosity η (γ̇). Appl Rheol. 2023;33(1):20220149. [10.1515/arh-2022-0149](https://doi.org/10.1515/arh-2022-0149).[Search in Google Scholar](https://scholar.google.com/scholar?q=Ellwanger%20F%2C%20Georgantopoulos%20CK%2C%20Karbstein%20HP%2C%20Wilhelm%20M%2C%20Azad%20Emin%20M.%20Application%20of%20the%20ramp%20test%20from%20a%20closed%20cavity%20rheometer%20to%20obtain%20the%20steady-state%20shear%20viscosity%20%CE%B7%20%28%CE%B3%CC%87%29.%20Appl%20Rheol.%202023%3B33%281%29%3A20220149.%2010.1515%2Farh-2022-0149%20.)

\[29\] Dunleavy Jr, JE, Middleman S. Correlation of shear behavior of solutions of polyisobutylene. Trans Soc Rheol. 1966;10(1):157–68. [10.1122/1.549055](https://doi.org/10.1122/1.549055).[Search in Google Scholar](https://scholar.google.com/scholar?q=Dunleavy%20Jr%2C%20JE%2C%20Middleman%20S.%20Correlation%20of%20shear%20behavior%20of%20solutions%20of%20polyisobutylene.%20Trans%20Soc%20Rheol.%201966%3B10%281%29%3A157%E2%80%9368.%2010.1122%2F1.549055%20.)

\[30\] Brewster RA, Irvine Jr TF. Similitude considerations in laminar flow of modified power law fluids in circular ducts. Waerme-Stoffuebertrag. Germany, Federal Republic of 1987;21(2/3):83–6. [10.1007/BF01377563](https://doi.org/10.1007/BF01377563).[Search in Google Scholar](https://scholar.google.com/scholar?q=Brewster%20RA%2C%20Irvine%20Jr%20TF.%20Similitude%20considerations%20in%20laminar%20flow%20of%20modified%20power%20law%20fluids%20in%20circular%20ducts.%20Waerme-Stoffuebertrag.%20Germany%2C%20Federal%20Republic%20of%201987%3B21%282%2F3%29%3A83%E2%80%936.%2010.1007%2FBF01377563%20.)

\[31\] Kristiawan B, Kamal S. A modified power law approach for rheological titania nanofluids flow behavior in a circular conduit. J Nanofluids. 2015;4(2):187–95. [10.1166/jon.2015.1139](https://doi.org/10.1166/jon.2015.1139).[Search in Google Scholar](https://scholar.google.com/scholar?q=Kristiawan%20B%2C%20Kamal%20S.%20A%20modified%20power%20law%20approach%20for%20rheological%20titania%20nanofluids%20flow%20behavior%20in%20a%20circular%20conduit.%20J%20Nanofluids.%202015%3B4%282%29%3A187%E2%80%9395.%2010.1166%2Fjon.2015.1139%20.)

\[32\] Thiébaud F. Determination of an innovative consistent law for the rheological behavior of polymer/carbon nanotubes composites. Soft Nanosci Lett. 2011;1(01):1–5. [10.4236/snl.2011.11001](https://doi.org/10.4236/snl.2011.11001).[Search in Google Scholar](https://scholar.google.com/scholar?q=Thi%C3%A9baud%20F.%20Determination%20of%20an%20innovative%20consistent%20law%20for%20the%20rheological%20behavior%20of%20polymer%2Fcarbon%20nanotubes%20composites.%20Soft%20Nanosci%20Lett.%202011%3B1%2801%29%3A1%E2%80%935.%2010.4236%2Fsnl.2011.11001%20.)

\[33\] Bitsch B, Dittmann J, Schmitt M, Scharfer P, Schabel W, Willenbacher N. A novel slurry concept for the fabrication of lithium-ion battery electrodes with beneficial properties. J Power Sources. 2014;265:81–90. [10.1016/j.jpowsour.2014.04.115](https://doi.org/10.1016/j.jpowsour.2014.04.115).[Search in Google Scholar](https://scholar.google.com/scholar?q=Bitsch%20B%2C%20Dittmann%20J%2C%20Schmitt%20M%2C%20Scharfer%20P%2C%20Schabel%20W%2C%20Willenbacher%20N.%20A%20novel%20slurry%20concept%20for%20the%20fabrication%20of%20lithium-ion%20battery%20electrodes%20with%20beneficial%20properties.%20J%20Power%20Sources.%202014%3B265%3A81%E2%80%9390.%2010.1016%2Fj.jpowsour.2014.04.115%20.)

**Received:** 2023-09-21

**Revised:** 2024-03-26

**Accepted:** 2024-03-29

**Published Online:** 2024-04-23

This work is licensed under the Creative Commons Attribution 4.0 International License.

## Articles in the same Issue

1. Research Articles
2. [Bearing behavior of pile foundation in karst region: Physical model test and finite element analysis](https://www.degruyterbrill.com/document/doi/10.1515/arh-2023-0115/html)
3. [Study on precursor information and disaster mechanism of sudden change of seepage in mining rock mass](https://www.degruyterbrill.com/document/doi/10.1515/arh-2023-0116/html)
4. Viscosity model based on Giesekus equation
5. [Two-dimensional rheo-optical measurement system to study dynamics and structure of complex fluids](https://www.degruyterbrill.com/document/doi/10.1515/arh-2024-0006/html)
6. [Assessment of heat transfer capabilities of some known nanofluids under turbulent flow conditions in a five-turn spiral pipe flow](https://www.degruyterbrill.com/document/doi/10.1515/arh-2024-0002/html)
7. [Cubic autocatalysis implementation in blood for non-Newtonian tetra hybrid nanofluid model through bounded artery](https://www.degruyterbrill.com/document/doi/10.1515/arh-2024-0007/html)
8. [Ramification of Hall effects in a non-Newtonian model past an inclined microchannel with slip and convective boundary conditions](https://www.degruyterbrill.com/document/doi/10.1515/arh-2024-0010/html)
9. [Computational analysis of nanoparticles and waste discharge concentration past a rotating sphere with Lorentz forces](https://www.degruyterbrill.com/document/doi/10.1515/arh-2024-0012/html)
10. [Viscoplastic fluid flow in pipes: A rheological study using *in-situ* laser Doppler velocimetry](https://www.degruyterbrill.com/document/doi/10.1515/arh-2024-0015/html)
11. [Prediction of sensory textures of cosmetics using large amplitude oscillatory shear and extensional rheology](https://www.degruyterbrill.com/document/doi/10.1515/arh-2024-0016/html)
12. [Effect of bell plate structure on high- and low-frequency characteristics of hydraulic mount](https://www.degruyterbrill.com/document/doi/10.1515/arh-2024-0014/html)
13. [Computational role of the heat transfer phenomenon in the reactive dynamics of catalytic nanolubricant flow past a horizontal microchannel](https://www.degruyterbrill.com/document/doi/10.1515/arh-2024-0017/html)
14. [Exploring concentration-dependent transport properties on an unsteady Riga plate by incorporating thermal radiation with activation energy and gyrotactic microorganisms](https://www.degruyterbrill.com/document/doi/10.1515/arh-2024-0019/html)
15. [Calendering of non-isothermal viscoelastic sheets of finite thickness: A theoretical study](https://www.degruyterbrill.com/document/doi/10.1515/arh-2024-0018/html)
16. [Electromagnetic control and heat transfer enhancement in exothermic reactions experiencing current density: The study preventing thermal explosions in reactive flow](https://www.degruyterbrill.com/document/doi/10.1515/arh-2024-0020/html)
17. [Characterization of the translational shear properties of the magnetorheological elastomers embedding the tilt chain-like structure](https://www.degruyterbrill.com/document/doi/10.1515/arh-2024-0022/html)
18. [Low-cost rolling ball viscometer for the evaluation of Newtonian and shear-thinning fluids](https://www.degruyterbrill.com/document/doi/10.1515/arh-2024-0021/html)
19. [Impact of calcination temperature, organic additive percentages, and testing temperature on the rheological behaviour of dried sewage sludge](https://www.degruyterbrill.com/document/doi/10.1515/arh-2024-0025/html)
20. [Rheo-NMR velocimetry of nanocrystalline cellulose suspensions](https://www.degruyterbrill.com/document/doi/10.1515/arh-2024-0026/html)
21. Review Articles
22. [Master curves construction for viscoelastic functions of bituminous materials](https://www.degruyterbrill.com/document/doi/10.1515/arh-2023-0117/html)
23. [Electrorheological characterization of complex fluids used in electrohydrodynamic processes: Technical issues and challenges](https://www.degruyterbrill.com/document/doi/10.1515/arh-2024-0024/html)
24. Corrigendum
25. [Corrigendum to: “The ductility performance of concrete using glass fiber mesh in beam specimens”](https://www.degruyterbrill.com/document/doi/10.1515/arh-2024-0199/html)
26. Special Issue on The rheological test, modeling and numerical simulation of rock material - Part I
27. [Study on the evolution of permeability properties of limestone under different stress paths](https://www.degruyterbrill.com/document/doi/10.1515/arh-2024-0003/html)
28. [Shale hydraulic fracture morphology and inter-well interference rule under multi-wellbore test](https://www.degruyterbrill.com/document/doi/10.1515/arh-2024-0005/html)
29. [Investigation and numerical simulation study on the vertical bearing mechanism of large-diameter overlength piles in water-enriched soft soil areas](https://www.degruyterbrill.com/document/doi/10.1515/arh-2024-0008/html)
30. [Evolution characteristics of calcareous sand force chain based on particle breakage](https://www.degruyterbrill.com/document/doi/10.1515/arh-2024-0009/html)
31. [Structural damage characteristics and mechanism of granite residual soil](https://www.degruyterbrill.com/document/doi/10.1515/arh-2024-0011/html)
32. [Rheological characteristics and seepage laws of sandstone specimens containing an inclined single fracture under three-dimensional stress](https://www.degruyterbrill.com/document/doi/10.1515/arh-2024-0013/html)

[^1]: (1)

where the zero-shear viscosity is given by and is the characteristic time. For *n* = 1, a Newtonian viscosity is resulted. When , shear thinning occurs, whereas shear thickening does for . As *n* approaches 0, shear thinning becomes severer. Let us introduce a dimensionless shear rate of the form:

[^2]: (2)

Using equations ([2](#j_arh-2024-0004_eq_002)) and ([1](#j_arh-2024-0004_eq_001)) can be written again as

[^3]: (4)

where

[^4]: (5)

where is the modified dimensionless shear rate in the Carreau model. It should be emphasized that *λ* designates the onset point at which the nonlinear viscous behavior starts. Consider that a fluid follows the ZFD model, which follows \[[1](#j_arh-2024-0004_ref_001 "[1] Carreau PJ, De Kee DC, Chhabra RP. Rheology of polymeric systems: principles and applications. München: Carl Hanser; 2021.10.3139/9781569907238.fmSearch in Google Scholar"),[22](#j_arh-2024-0004_ref_022 "[22] Giesekus H. A simple constitutive equation for polymer fluids based on the concept of deformation-dependent tensorial mobility. J Non-Newtonian Fluid Mech. 1982;11(1–2):69–109. 10.1016/0377-0257(82)85016-7.Search in Google Scholar")\]

[^5]: (7)

Using the aforementioned expression, equation ([4](#j_arh-2024-0004_eq_004)) can be written again as

[^6]: (8)

This is the way the Carreau viscosity model is related to the ZFD equation.

[^7]: (10)

where

[^8]: (13)

Interestingly, a continued fraction derived from equation ([7](#j_arh-2024-0004_eq_007)) denoted as

[^9]: (15)

Introducing a truncation term , the viscosity becomes a five-parameter model \[[27](#j_arh-2024-0004_ref_027 "[27] Carreau PJ, Kee DD, Daroux M. An analysis of the viscous behaviour of polymeric solutions. Can J Chem Eng. 1979;57(2):135–40. 10.1002/cjce.5450570202.Search in Google Scholar")\]:

[^10]: (16)

[^11]: (17)

where the coefficients have been obtained by fitting equations ([15](#j_arh-2024-0004_eq_015)) to ([4](#j_arh-2024-0004_eq_004)).

[^12]: (18)

For *a* = 2, is attained. Substituting equation ([18](#j_arh-2024-0004_eq_018)) into equation ([4](#j_arh-2024-0004_eq_004)), we have the Carreau–Yasuda model expressed as

[^13]: (19)

Let us incorporate the Yasuda-type index to equation ([10](#j_arh-2024-0004_eq_010)):

[^14]: (21)

where is the infinite-shear viscosity. This work has explained it by equation ([18](#j_arh-2024-0004_eq_018)) but the index *a* was initially employed to simply compensate error present in the Carreau model near the critical shear rate \[[7](#j_arh-2024-0004_ref_007 "[7] Yasuda KY, Armstrong RC, Cohen RE. Shear flow properties of concentrated solutions of linear and star branched polystyrenes. Rheol Acta. 1981;20(2):163–78. 10.1007/BF01513059.Search in Google Scholar")\]. As Yasuda suggested, the model presented in equation ([15](#j_arh-2024-0004_eq_015)) can also be used to control the curvature at the inflection point by adjusting parameter *a,* as shown in [Figure 8](#j_arh-2024-0004_fig_008). As *a* increases, the model well follows the truncated power-law model. At *a* = 8, the model almost realizes the sharp edge at Γ = 1. However, it should be mentioned that finding a clear physical interpretation for this parameter beyond its utility in fitting experimental data remains challenging. Therefore, it is considered more appropriate to vary the mobility parameter alpha while keeping *a* at a value of 2.

[^15]: \[1\] Carreau PJ, De Kee DC, Chhabra RP. Rheology of polymeric systems: principles and applications. München: Carl Hanser; 2021.[10.3139/9781569907238.fm](https://doi.org/10.3139/9781569907238.fm) [Search in Google Scholar](https://scholar.google.com/scholar?q=Carreau%20PJ%2C%20De%20Kee%20DC%2C%20Chhabra%20RP.%20Rheology%20of%20polymeric%20systems%3A%20principles%20and%20applications.%20M%C3%BCnchen%3A%20Carl%20Hanser%3B%202021.)

[^16]: \[2\] Osswald T, Rudolph N. Polymer rheology. München: Carl Hanser; 2015.[10.1007/978-1-56990-523-4](https://doi.org/10.1007/978-1-56990-523-4) [Search in Google Scholar](https://scholar.google.com/scholar?q=Osswald%20T%2C%20Rudolph%20N.%20Polymer%20rheology.%20M%C3%BCnchen%3A%20Carl%20Hanser%3B%202015.)

[^17]: \[3\] Cross MM. Rheology of non-newtonian fluids: a new flow equation for pseudoplastic systems. J Colloid Sci. 1965;20(5):417–37. [10.1016/0095-8522(65)90022-X](https://doi.org/10.1016/0095-8522\(65\)90022-X).[Search in Google Scholar](https://scholar.google.com/scholar?q=Cross%20MM.%20Rheology%20of%20non-newtonian%20fluids%3A%20a%20new%20flow%20equation%20for%20pseudoplastic%20systems.%20J%20Colloid%20Sci.%201965%3B20%285%29%3A417%E2%80%9337.%2010.1016%2F0095-8522%2865%2990022-X%20.)

[^18]: \[4\] Cross MM. Polymer rheology: influence of molecular weight and polydispersity. J Appl Polym Sci. 1969;13(4):765–74. [10.1002/app.1969.070130415](https://doi.org/10.1002/app.1969.070130415).[Search in Google Scholar](https://scholar.google.com/scholar?q=Cross%20MM.%20Polymer%20rheology%3A%20influence%20of%20molecular%20weight%20and%20polydispersity.%20J%20Appl%20Polym%20Sci.%201969%3B13%284%29%3A765%E2%80%9374.%2010.1002%2Fapp.1969.070130415%20.)

[^19]: \[5\] Carreau PJ. Rheological equations from molecular network theories. Trans Soc Rheology. 1972;16(1):99–127. [10.1122/1.549276](https://doi.org/10.1122/1.549276).[Search in Google Scholar](https://scholar.google.com/scholar?q=Carreau%20PJ.%20Rheological%20equations%20from%20molecular%20network%20theories.%20Trans%20Soc%20Rheology.%201972%3B16%281%29%3A99%E2%80%93127.%2010.1122%2F1.549276%20.)

[^20]: \[6\] Yasuda K. Investigation of the analogies between viscometric and linear viscoelastic properties of polystyrene fluids \[dissertation\]. Cambridge (MA): Massachusetts Institute of Technology; 1979. https://dspace.mit.edu/handle/1721.1/16043.[Search in Google Scholar](https://scholar.google.com/scholar?q=Yasuda%20K.%20Investigation%20of%20the%20analogies%20between%20viscometric%20and%20linear%20viscoelastic%20properties%20of%20polystyrene%20fluids%20%5Bdissertation%5D.%20Cambridge%20%28MA%29%3A%20Massachusetts%20Institute%20of%20Technology%3B%201979.%20https%3A%2F%2Fdspace.mit.edu%2Fhandle%2F1721.1%2F16043.)

[^21]: \[7\] Yasuda KY, Armstrong RC, Cohen RE. Shear flow properties of concentrated solutions of linear and star branched polystyrenes. Rheol Acta. 1981;20(2):163–78. [10.1007/BF01513059](https://doi.org/10.1007/BF01513059).[Search in Google Scholar](https://scholar.google.com/scholar?q=Yasuda%20KY%2C%20Armstrong%20RC%2C%20Cohen%20RE.%20Shear%20flow%20properties%20of%20concentrated%20solutions%20of%20linear%20and%20star%20branched%20polystyrenes.%20Rheol%20Acta.%201981%3B20%282%29%3A163%E2%80%9378.%2010.1007%2FBF01513059%20.)

[^22]: \[8\] Kim SK. Flow-rate based method for velocity of fully developed laminar flow in tubes. J Rheol. 2018 Nov 1;62(6):1397–407. [10.1122/1.5041958](https://doi.org/10.1122/1.5041958).[Search in Google Scholar](https://scholar.google.com/scholar?q=Kim%20SK.%20Flow-rate%20based%20method%20for%20velocity%20of%20fully%20developed%20laminar%20flow%20in%20tubes.%20J%20Rheol.%202018%20Nov%201%3B62%286%29%3A1397%E2%80%93407.%2010.1122%2F1.5041958%20.)

[^23]: \[9\] Kim SK. Flow rate based framework for solving viscoplastic flow with slip. J Non-Newtonian Fluid Mech. 2019;269:37–46. [10.1016/j.jnnfm.2019.06.002](https://doi.org/10.1016/j.jnnfm.2019.06.002).[Search in Google Scholar](https://scholar.google.com/scholar?q=Kim%20SK.%20Flow%20rate%20based%20framework%20for%20solving%20viscoplastic%20flow%20with%20slip.%20J%20Non-Newtonian%20Fluid%20Mech.%202019%3B269%3A37%E2%80%9346.%2010.1016%2Fj.jnnfm.2019.06.002%20.)

[^24]: \[10\] Kim SK, Kazmer DO, Colon AR, Coogan TJ, Peterson AM. Non-Newtonian modeling of contact pressure in fused filament fabrication. J Rheol. 2021;65(1):27–42. [10.1122/8.0000052](https://doi.org/10.1122/8.0000052).[Search in Google Scholar](https://scholar.google.com/scholar?q=Kim%20SK%2C%20Kazmer%20DO%2C%20Colon%20AR%2C%20Coogan%20TJ%2C%20Peterson%20AM.%20Non-Newtonian%20modeling%20of%20contact%20pressure%20in%20fused%20filament%20fabrication.%20J%20Rheol.%202021%3B65%281%29%3A27%E2%80%9342.%2010.1122%2F8.0000052%20.)

[^25]: \[11\] Hong J, Kim SK, Cho YH. Flow and solidification of semi-crystalline polymer during micro-injection molding. Int J Heat Mass Transf. 2020;153:119576. [10.1016/j.ijheatmasstransfer.2020.119576](https://doi.org/10.1016/j.ijheatmasstransfer.2020.119576) [Search in Google Scholar](https://scholar.google.com/scholar?q=Hong%20J%2C%20Kim%20SK%2C%20Cho%20YH.%20Flow%20and%20solidification%20of%20semi-crystalline%20polymer%20during%20micro-injection%20molding.%20Int%20J%20Heat%20Mass%20Transf.%202020%3B153%3A119576.%2010.1016%2Fj.ijheatmasstransfer.2020.119576)

[^26]: \[12\] Kim SK, Kazmer DO. Non-isothermal non-Newtonian three-dimensional flow simulation of fused filament fabrication. Addit Manuf. 2022;55:102833. [10.1016/j.addma.2022.102833](https://doi.org/10.1016/j.addma.2022.102833).[Search in Google Scholar](https://scholar.google.com/scholar?q=Kim%20SK%2C%20Kazmer%20DO.%20Non-isothermal%20non-Newtonian%20three-dimensional%20flow%20simulation%20of%20fused%20filament%20fabrication.%20Addit%20Manuf.%202022%3B55%3A102833.%2010.1016%2Fj.addma.2022.102833%20.)

[^27]: \[13\] Kim SK. Collective viscosity model for shear thinning polymeric materials. Rheol Acta. 2020;59(1):63–72. [10.1007/s00397-019-01180-w](https://doi.org/10.1007/s00397-019-01180-w).[Search in Google Scholar](https://scholar.google.com/scholar?q=Kim%20SK.%20Collective%20viscosity%20model%20for%20shear%20thinning%20polymeric%20materials.%20Rheol%20Acta.%202020%3B59%281%29%3A63%E2%80%9372.%2010.1007%2Fs00397-019-01180-w%20.)

[^28]: \[14\] Kazmer DO, Colon AR, Peterson AM, Kim SK. Concurrent characterization of compressibility and viscosity in extrusion-based additive manufacturing of acrylonitrile butadiene styrene with fault diagnoses. Addit Manuf. 2021;46:102106. [10.1016/j.addma.2021.102106](https://doi.org/10.1016/j.addma.2021.102106).[Search in Google Scholar](https://scholar.google.com/scholar?q=Kazmer%20DO%2C%20Colon%20AR%2C%20Peterson%20AM%2C%20Kim%20SK.%20Concurrent%20characterization%20of%20compressibility%20and%20viscosity%20in%20extrusion-based%20additive%20manufacturing%20of%20acrylonitrile%20butadiene%20styrene%20with%20fault%20diagnoses.%20Addit%20Manuf.%202021%3B46%3A102106.%2010.1016%2Fj.addma.2021.102106%20.)

[^29]: \[15\] Kim SK, Jeong A. Numerical simulation of crystal growth in injection molded thermoplastics based on Monte Carlo method with shear rate tracking. Int J Precis Eng Manuf. 2019;20:641–50. [10.1007/s12541-019-00089-x](https://doi.org/10.1007/s12541-019-00089-x).[Search in Google Scholar](https://scholar.google.com/scholar?q=Kim%20SK%2C%20Jeong%20A.%20Numerical%20simulation%20of%20crystal%20growth%20in%20injection%20molded%20thermoplastics%20based%20on%20Monte%20Carlo%20method%20with%20shear%20rate%20tracking.%20Int%20J%20Precis%20Eng%20Manuf.%202019%3B20%3A641%E2%80%9350.%2010.1007%2Fs12541-019-00089-x%20.)

[^30]: \[16\] Jung JS, Kim SK. Rapid numerical estimation of pressure drop in hot runner system. Micromachines. 2021;12(2):207. [10.3390/mi12020207](https://doi.org/10.3390/mi12020207).[Search in Google Scholar](https://scholar.google.com/scholar?q=Jung%20JS%2C%20Kim%20SK.%20Rapid%20numerical%20estimation%20of%20pressure%20drop%20in%20hot%20runner%20system.%20Micromachines.%202021%3B12%282%29%3A207.%2010.3390%2Fmi12020207%20.) [PubMed](https://pubmed.ncbi.nlm.nih.gov/33670694/) [PubMed Central](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7922069/)

[^31]: \[17\] Tadmor Z, Gogos CG. Principles of polymer processing. New York: John Wiley & Sons; 2013.[Search in Google Scholar](https://scholar.google.com/scholar?q=Tadmor%20Z%2C%20Gogos%20CG.%20Principles%20of%20polymer%20processing.%20New%20York%3A%20John%20Wiley%20%26%20Sons%3B%202013.)

[^32]: \[18\] DeWitt TW. A rheological equation of state which predicts non‐newtonian viscosity, normal stresses, and dynamic moduli. J Appl Phys. 1955;26(7):889–94. [10.1063/1.1722114](https://doi.org/10.1063/1.1722114).[Search in Google Scholar](https://scholar.google.com/scholar?q=DeWitt%20TW.%20A%20rheological%20equation%20of%20state%20which%20predicts%20non%E2%80%90newtonian%20viscosity%2C%20normal%20stresses%2C%20and%20dynamic%20moduli.%20J%20Appl%20Phys.%201955%3B26%287%29%3A889%E2%80%9394.%2010.1063%2F1.1722114%20.)

[^33]: \[19\] Bird RB, Carreau PJ. A nonlinear viscoelastic model for polymer solutions and melts – I. Chem Eng Sci. 1968;23(5):427–34. [10.1016/0009-2509(68)87018-6](https://doi.org/10.1016/0009-2509\(68\)87018-6).[Search in Google Scholar](https://scholar.google.com/scholar?q=Bird%20RB%2C%20Carreau%20PJ.%20A%20nonlinear%20viscoelastic%20model%20for%20polymer%20solutions%20and%20melts%20%E2%80%93%20I.%20Chem%20Eng%20Sci.%201968%3B23%285%29%3A427%E2%80%9334.%2010.1016%2F0009-2509%2868%2987018-6%20.)

[^34]: \[20\] Cho YI, Kensey KR. Effects of the non-newtonian viscosity of blood on flows in a diseased arterial vessel. Part 1: Steady flows. Biorheology. 1991;28(3-4):241–62. [10.3233/BIR-1991-283-415](https://doi.org/10.3233/BIR-1991-283-415).[Search in Google Scholar](https://scholar.google.com/scholar?q=Cho%20YI%2C%20Kensey%20KR.%20Effects%20of%20the%20non-newtonian%20viscosity%20of%20blood%20on%20flows%20in%20a%20diseased%20arterial%20vessel.%20Part%201%3A%20Steady%20flows.%20Biorheology.%201991%3B28%283-4%29%3A241%E2%80%9362.%2010.3233%2FBIR-1991-283-415%20.)

[^35]: \[21\] Kelly NS, Gill HS, Cookson AN, Fraser KH. Influence of shear-thinning blood rheology on the laminar-turbulent transition over a backward facing step. Fluids. 2020;5(2):57. [10.3390/fluids5020057](https://doi.org/10.3390/fluids5020057).[Search in Google Scholar](https://scholar.google.com/scholar?q=Kelly%20NS%2C%20Gill%20HS%2C%20Cookson%20AN%2C%20Fraser%20KH.%20Influence%20of%20shear-thinning%20blood%20rheology%20on%20the%20laminar-turbulent%20transition%20over%20a%20backward%20facing%20step.%20Fluids.%202020%3B5%282%29%3A57.%2010.3390%2Ffluids5020057%20.)

[^36]: \[22\] Giesekus H. A simple constitutive equation for polymer fluids based on the concept of deformation-dependent tensorial mobility. J Non-Newtonian Fluid Mech. 1982;11(1–2):69–109. [10.1016/0377-0257(82)85016-7](https://doi.org/10.1016/0377-0257\(82\)85016-7).[Search in Google Scholar](https://scholar.google.com/scholar?q=Giesekus%20H.%20A%20simple%20constitutive%20equation%20for%20polymer%20fluids%20based%20on%20the%20concept%20of%20deformation-dependent%20tensorial%20mobility.%20J%20Non-Newtonian%20Fluid%20Mech.%201982%3B11%281%E2%80%932%29%3A69%E2%80%93109.%2010.1016%2F0377-0257%2882%2985016-7%20.)

[^37]: \[23\] Schaible T, Bonten C. In-line measurement and modeling of temperature, pressure, and blowing agent dependent viscosity of polymer melts. Appl Rheol. 2022;32(1):69–82. [10.1515/arh-2022-0123](https://doi.org/10.1515/arh-2022-0123).[Search in Google Scholar](https://scholar.google.com/scholar?q=Schaible%20T%2C%20Bonten%20C.%20In-line%20measurement%20and%20modeling%20of%20temperature%2C%20pressure%2C%20and%20blowing%20agent%20dependent%20viscosity%20of%20polymer%20melts.%20Appl%20Rheol.%202022%3B32%281%29%3A69%E2%80%9382.%2010.1515%2Farh-2022-0123%20.)

[^38]: \[24\] Han CD. Rheology and processing of polymeric materials. Polymer rheology. Vol. 1, New York: Oxford University Press; 2007.[10.1093/oso/9780195187823.001.0001](https://doi.org/10.1093/oso/9780195187823.001.0001) [Search in Google Scholar](https://scholar.google.com/scholar?q=Han%20CD.%20Rheology%20and%20processing%20of%20polymeric%20materials.%20Polymer%20rheology.%20Vol.%201%2C%20New%20York%3A%20Oxford%20University%20Press%3B%202007.)

[^39]: \[25\] Bird RB, Wiest JM. Constitutive equations for polymeric liquids. Annu Rev fluid Mech. 1995;27(1):169–93. [10.1146/annurev.fl.27.010195.001125](https://doi.org/10.1146/annurev.fl.27.010195.001125).[Search in Google Scholar](https://scholar.google.com/scholar?q=Bird%20RB%2C%20Wiest%20JM.%20Constitutive%20equations%20for%20polymeric%20liquids.%20Annu%20Rev%20fluid%20Mech.%201995%3B27%281%29%3A169%E2%80%9393.%2010.1146%2Fannurev.fl.27.010195.001125%20.)

[^40]: \[26\] Wiest JM, Bird RB. Molecular extension from the Giesekus model. J non-Newtonian fluid Mech. 1986;22(1):115–9. [10.1016/0377-0257(86)80007-6](https://doi.org/10.1016/0377-0257\(86\)80007-6).[Search in Google Scholar](https://scholar.google.com/scholar?q=Wiest%20JM%2C%20Bird%20RB.%20Molecular%20extension%20from%20the%20Giesekus%20model.%20J%20non-Newtonian%20fluid%20Mech.%201986%3B22%281%29%3A115%E2%80%939.%2010.1016%2F0377-0257%2886%2980007-6%20.)

[^41]: \[27\] Carreau PJ, Kee DD, Daroux M. An analysis of the viscous behaviour of polymeric solutions. Can J Chem Eng. 1979;57(2):135–40. [10.1002/cjce.5450570202](https://doi.org/10.1002/cjce.5450570202).[Search in Google Scholar](https://scholar.google.com/scholar?q=Carreau%20PJ%2C%20Kee%20DD%2C%20Daroux%20M.%20An%20analysis%20of%20the%20viscous%20behaviour%20of%20polymeric%20solutions.%20Can%20J%20Chem%20Eng.%201979%3B57%282%29%3A135%E2%80%9340.%2010.1002%2Fcjce.5450570202%20.)

[^42]: \[28\] Ellwanger F, Georgantopoulos CK, Karbstein HP, Wilhelm M, Azad Emin M. Application of the ramp test from a closed cavity rheometer to obtain the steady-state shear viscosity η (γ̇). Appl Rheol. 2023;33(1):20220149. [10.1515/arh-2022-0149](https://doi.org/10.1515/arh-2022-0149).[Search in Google Scholar](https://scholar.google.com/scholar?q=Ellwanger%20F%2C%20Georgantopoulos%20CK%2C%20Karbstein%20HP%2C%20Wilhelm%20M%2C%20Azad%20Emin%20M.%20Application%20of%20the%20ramp%20test%20from%20a%20closed%20cavity%20rheometer%20to%20obtain%20the%20steady-state%20shear%20viscosity%20%CE%B7%20%28%CE%B3%CC%87%29.%20Appl%20Rheol.%202023%3B33%281%29%3A20220149.%2010.1515%2Farh-2022-0149%20.)

[^43]: \[29\] Dunleavy Jr, JE, Middleman S. Correlation of shear behavior of solutions of polyisobutylene. Trans Soc Rheol. 1966;10(1):157–68. [10.1122/1.549055](https://doi.org/10.1122/1.549055).[Search in Google Scholar](https://scholar.google.com/scholar?q=Dunleavy%20Jr%2C%20JE%2C%20Middleman%20S.%20Correlation%20of%20shear%20behavior%20of%20solutions%20of%20polyisobutylene.%20Trans%20Soc%20Rheol.%201966%3B10%281%29%3A157%E2%80%9368.%2010.1122%2F1.549055%20.)

[^44]: \[30\] Brewster RA, Irvine Jr TF. Similitude considerations in laminar flow of modified power law fluids in circular ducts. Waerme-Stoffuebertrag. Germany, Federal Republic of 1987;21(2/3):83–6. [10.1007/BF01377563](https://doi.org/10.1007/BF01377563).[Search in Google Scholar](https://scholar.google.com/scholar?q=Brewster%20RA%2C%20Irvine%20Jr%20TF.%20Similitude%20considerations%20in%20laminar%20flow%20of%20modified%20power%20law%20fluids%20in%20circular%20ducts.%20Waerme-Stoffuebertrag.%20Germany%2C%20Federal%20Republic%20of%201987%3B21%282%2F3%29%3A83%E2%80%936.%2010.1007%2FBF01377563%20.)

[^45]: \[31\] Kristiawan B, Kamal S. A modified power law approach for rheological titania nanofluids flow behavior in a circular conduit. J Nanofluids. 2015;4(2):187–95. [10.1166/jon.2015.1139](https://doi.org/10.1166/jon.2015.1139).[Search in Google Scholar](https://scholar.google.com/scholar?q=Kristiawan%20B%2C%20Kamal%20S.%20A%20modified%20power%20law%20approach%20for%20rheological%20titania%20nanofluids%20flow%20behavior%20in%20a%20circular%20conduit.%20J%20Nanofluids.%202015%3B4%282%29%3A187%E2%80%9395.%2010.1166%2Fjon.2015.1139%20.)

[^46]: \[32\] Thiébaud F. Determination of an innovative consistent law for the rheological behavior of polymer/carbon nanotubes composites. Soft Nanosci Lett. 2011;1(01):1–5. [10.4236/snl.2011.11001](https://doi.org/10.4236/snl.2011.11001).[Search in Google Scholar](https://scholar.google.com/scholar?q=Thi%C3%A9baud%20F.%20Determination%20of%20an%20innovative%20consistent%20law%20for%20the%20rheological%20behavior%20of%20polymer%2Fcarbon%20nanotubes%20composites.%20Soft%20Nanosci%20Lett.%202011%3B1%2801%29%3A1%E2%80%935.%2010.4236%2Fsnl.2011.11001%20.)

[^47]: \[33\] Bitsch B, Dittmann J, Schmitt M, Scharfer P, Schabel W, Willenbacher N. A novel slurry concept for the fabrication of lithium-ion battery electrodes with beneficial properties. J Power Sources. 2014;265:81–90. [10.1016/j.jpowsour.2014.04.115](https://doi.org/10.1016/j.jpowsour.2014.04.115).[Search in Google Scholar](https://scholar.google.com/scholar?q=Bitsch%20B%2C%20Dittmann%20J%2C%20Schmitt%20M%2C%20Scharfer%20P%2C%20Schabel%20W%2C%20Willenbacher%20N.%20A%20novel%20slurry%20concept%20for%20the%20fabrication%20of%20lithium-ion%20battery%20electrodes%20with%20beneficial%20properties.%20J%20Power%20Sources.%202014%3B265%3A81%E2%80%9390.%2010.1016%2Fj.jpowsour.2014.04.115%20.)