# Multi-Objective Loss Balancing for Physics-Informed Deep Learning

Rafael Bischof Swiss Data Science Center Zürich, Switzerland rafael.bischof@sdsc.ethz.ch 

Michael A. Kraus Chair of Concrete Structures and Bridge Design (IBK)/ Design++, ETH Zürich Zürich, Switzerland kraus@ibk.baug.ethz.ch 

Abstract—Physics-Informed Neural Networks (PINN) are algorithms from deep learning leveraging physical laws by including partial differential equations together with a respective set of boundary and initial conditions as penalty terms into their loss function. In this work, we observe the significant role of correctly weighting the combination of multiple competitive loss functions for training PINNs effectively. To this end, we implement and evaluate different methods aiming at balancing the contributions of multiple terms of the PINNs loss function and their gradients. After reviewing of three existing loss scaling approaches (Learning Rate Annealing, GradNorm and SoftAdapt), we propose a novel self-adaptive loss balancing scheme for PINNs named ReLoBRaLo (Relative Loss Balancing with Random Lookback). We extensively evaluate the performance of the aforementioned balancing schemes by solving both forward as well as inverse problems on three benchmark PDEs for PINNs: Burgers’ equation, Kirchhoff’s plate bending equation and Helmholtz’s equation. The results show that ReLoBRaLo is able to consistently outperform the baseline of existing scaling methods in terms of accuracy, while also inducing significantly less computational overhead. 

## I. Introduction

The emergence of Physics-Informed Neural Networks (PINNs) [48] has sparked a lot of interest in domains that see themselves regularly confronted with problems in the low data regime. By leveraging well-known physical laws and incorporating them as implicit prior into the deeplearning pipeline, PINNs were shown to require little to no data in order to approximate partial differential equations (PDE) of varying complexity [18, 48]. 

We consider the case where PINNs are used for finding the unknown, underlying function proper to solve a parameterised PDE. PDEs generally consist of a governing equation and a set of boundary as well as initial conditions. 

The authors would like to thankfully acknowledge the facilities of Design++ at ETH Zürich and the funding through ETH Foundation grant No. 2020-HS-388 (provided by Kollbrunner/Rodio) as well as the SDSC Project "Domain-Aware AI-augmented Design of Bridges (DAAAD Bridges)". 

When trained jointly, i.e. as multi-objective optimisation (MOO), these equations form a set of objective functions that guide the model to approximate a function that satisfies the PDE. While several established and wellstudied numerical methods already exist for addressing this problem, such as the Finite Element Method (FEM), Finite Difference Method or Wavelets resp. Laplace transform methods [1, 17, 22, 57], PINNs offer significant advantages, such as being end-to-end differentiable, mesh-free and avoiding the curse of dimensionality [16, 46]. They could therefore prove useful in several engineering applications, such as inversion and surrogate modeling in solid mechanics [19], design optimisation [38] or structural health monitoring and system identification [69]. PINNs have also been successfully applied in computational fluid mechanics and dynamics for surrogate modelling of numerically expensive fluid flow simulations [68], identification of hidden quantities of interest (velocity, pressure) from spatio-temporal visualisations of a passive scaler (dye or smoke) [50] or in an inverse heat transfer application setting in flow past a cylinder without thermal boundaries [4]. 

However, further research is necessary to tackle current failure modes [39, 66], one of which is the issue of gradient pathologies arising from imbalanced loss terms during training [65]. With the various terms in the objective function stemming from physical laws, they are naturally bound to units of measurements that can vary significantly in magnitude. Consequently, the signal strengths of backpropagated gradients might differ from term to term and lead to pathologies that were shown to impede proper training and cause imbalanced solutions [65], hence posing challenges to global optimisation methods such as Adam, Stochastic Gradient Descent (SGD), or L-BFGS [28, 31, 62, 70]. As a counter-measure, every individual term i may be scaled by a factor $\lambda _ { i }$ in order to balance its contribution to the total gradient. However, manual tuning of these scaling factors requires laborious grid search and becomes intractable as the number of terms grows. 

This work investigates different schemes aiming at adaptively balancing the contributions of multiple terms and their gradients in the loss function by selecting the optimal scaling factors $\lambda _ { i }$ in order to improve approximation capabilities of PINNs. To this end, we compare the effectiveness of Learning Rate Annealing (LRAnnealing) [65], proposed in the context of PINNs, to two approaches originating from Computer Vision applications: GradNorm [8] and SoftAdapt [20]. In addition, we derive and present our own variation of an adaptive loss scaling technique, ReLoBRaLo (Relative Loss Balancing with Random Lookback), that we found to be more effective at similar efficiency compared to state-of-the-art by testing the algorithms on various benchmark problems for PINNs in the forward and inverse setting: Helmholtz, Burgers and Kirchhoff PDEs. 

This paper is organised as follows: we first provide a short introduction to the problem as well as the state-of-theart of PINNs in sec. III. Further methodical background on multi-objective optimisation (MOO), the framework of Physics-Informed Neural Networks (PINNs) and loss balancing for PINNs training is presented in sec. V. In sec. VI we introduce ReLoBRaLo as a novel selfadaptive loss balancing method. Sec. VIII reports numerical results of the developed approach against state-of-theart methods for several examples in the forward and inverse setting: Burgers, Kirchhoff and Helmholtz PDEs. Sec. IX presents results of ablation studies as well as a discussion of findings and drawing further conclusions on ReLoBRaLo and its hyperparameter settings across all examples of this paper. Finally, a summary together with an outlook is given in sec. X. All code produced within this publication is freely available and open access here: https://github.com/rbischof/relative_balancing. 

## II. State-of-the-Art and Related Work

Using neural networks to approximate the solutions of Ordinary Differential Equations (ODEs) and PDEs has been the subject of several studies over the past decade. Initially, Lagaris et al. trained neural networks to solve ODEs and PDEs on a predefined set of grid points [34, 35], while Sirignano proposed a method for solving high-dimensional PDEs through approximation of the solution by a neural network and especially emphasising training efficiency by incorporating mini-batch sampling in high dimensional settings compared to the computationally intractable finite mesh-based schemes [56]. Raissi et al. introduced the term Physics-Informed Neural Networks (PINN) and provided empirical justification by numerical simulations for a variety of nonlinear PDEs, including the Navier–Stokes equation and the Burgers equation [47]. Shin et al. provided first theoretical justification for PINNs by demonstrating convergence of linear elliptic and parabolic PDEs in the L2 sense [54]. 

However, PINN training efficiency, convergence, and accuracy remain serious challenges [49, 68]. Current research may be ordered into four main approaches: modifying structure of the NN, divide-and-conquer/domain decomposition, parameter initialisation and loss balancing. 

Jagtap et al. adapted the typical NN architecture to PINNs by introducing parameters that scale the input to the activation functions and get updated alongside the network’s parameters θ through gradient descent [25, 26]. The authors showed that the adaptive activation function significantly accelerated convergence and also improved solution accuracy. Kim et al. presented a fast and accurate PINN ROM with a nonlinear manifold solution representation, where the NN structure included an encoder and a decoder part [30]. Furthermore, a shallow masked encoder was trained using data from the full-order model simulations in order to use the trained decoder as representation of the nonlinear manifold solution. Peng et al. proposed dictionary-based PINNs to store and retrieve features and speed up convergence by merging prior information into the structure of NNs [45]. 

Other research focused on decomposing the computational domain in order to accelerate convergence. Jagtap et al. proposed conservative PINNs and extended PINNs that decompose the computational domain into several discrete sub-domains, each one solved independently using a separate, shallow PINN [23, 24]. Inspired by this work, Shukla et al. derived and investigated a distributed training framework for PINNs that used domain decomposition methods in both space and time-space [55]. To accelerate convergence, the distributed framework combined the benefits of conservative and extended PINNs. The timespace domain may become very large when solving PDEs with long time integration, causing the training cost of NNs to become extremely expensive. To that end, Meng et al. proposed a parareal PINN to address the longstanding issue [40]. The authors decomposed the long-time domain into many discrete short-time domains using a fast coarse-grained solver. Training multiple PINNs with many small data sets was much faster than training a single PINN with a large data set. For PDEs with longtime integration, the parareal PINN achieved a significant speedup. Kharazmi et al. introduced hp-variational PINNs to divide the computational space into the trial space and test space by combining domain decomposition and projection onto high-order polynomials [29]. A soft split of the problem domain incorporating variants of the Mixtureof-Experts approach were investigated by [2]. 

In most works, researchers resort to the Xavier initialisation [14] for selecting the PINN’s initial weights and biases. The effects of using more refined initialisation procedures has recently been gaining attention, with Liu et al. showing that a good initialisation can provide PINNs with a head start, allowing them to achieve fast convergence and improved accuracy [37]. Transfer learning for PINNs was introduced by Chakraborty et al. [6] and Goswami et al. [15] to initialise PINNs for dealing with multi-fidelity problems and brittle fracture problems, respectively. After their success in other fields of Deep Learning, meta-learning algorithms have also been implemented in the context of PINNs [12, 13, 51, 58], with Model-Agnostic Meta-Learning (MAML) being amongst the most popular ones [11]. Its second-order objective is to find an initialisation that is sub-optimal in itself, but from where the network requires only few labeled training samples and optimisation steps in order to specialise on a task and achieve high accuracy (few-shot learning). Subsequently, Nichol et al. proposed the REPTILE algorithm, which turns the second-order optimisation of MAML into a first-order approximation and therefore requires significantly less computation and memory while achieving similar performance [42]. Liu et al. applied the REPTILE algorithm to PINNs by regarding modifications of PDE parameters as separate tasks [37]. The resulting initialisation is such that the PINN converges in just a few optimisation steps for any choice of PDE parameters. 

Using derivative information of the target function during training of a neural network was introduced by Czarnecki et al. under the term Sobolev Training [9]. Sobolev Training proved to be more efficient in many applicable fields due to lower sample complexity compared to regular training. Son et al. enhance the concept of Sobolev Training in the strict mathematical sense using Sobolev norms in loss functions of neural networks for solving PDEs [61]. It was found that these novel Sobolev loss functions lead to significantly faster convergence on investigated examples compared to traditional L2 loss functions. NNs were used in plain as well as a Sobolev Training manner for constitutive modelling, where it was shown that mechanical relations can be seen as Sobolev training to successfully encapsulate several aspects of the constitutive behavior, such as strainstress-relationships arising from derivatives of a Helmholtz potential in hyperelasticity [32, 33, 63, 64]. 

Colby et al. observed that a weighted scalarisation of the multiple loss functions, defined by the sampled data and physical laws for PINNs training, plays a significant role for convergence [67]. Wang et al. recently published a Learning Rate Annealing algorithm that employs back-propagated gradient statistics in the training procedure in order to adaptively balance the terms’ contributions to the final loss [65] and investigated the issue of vanishing and exploding gradients that currently limits the applicability of PINNs [66]. To that end, the authors introduced a Neural Tangent Kernel (NTK), which appropriately assigns weights to each loss term at subtle performance improvement, in order to comprehend the training process for PINNs. Shin et al. developed the Lipschitz regularised loss for solving linear second-order elliptic and parabolic type PDEs [54]. McClenny et al. proposed a method for updating the adaptation weights in the loss function in relation to network parameters [39]. 

## III. Physics-Informed Neural Networks (PINNs)

This section reviews basic Physics-Informed Neural Networks (PINNs) concepts and recent developments. 

Consider the following abstract parameterised and non-

linear PDE problem: 

$$
\mathrm{PDE}: \mathcal {F} \left(\hat {\mathbf {u}}, \frac {\partial \hat {\mathbf {u}}}{\partial t}, \frac {\partial \hat {\mathbf {u}}}{\partial \mathbf {x}}, \dots ; \mu\right) = 0, \quad \mathbf {x} \in \Omega , t \in \Upsilon
$$

$$
\text { B   .   C   . }: \mathcal {B} \left(\hat {\mathbf {u}}, \frac {\partial \hat {\mathbf {u}}}{\partial \mathbf {x}}, \frac {\partial^ {2} \hat {\mathbf {u}}}{\partial \mathbf {x} ^ {2}}, \dots\right) = 0, \quad \mathbf {x} \in \Gamma \tag {1}
$$

$$
\mathrm{I.C.}: \mathcal {C} \left(\hat {\mathbf {u}}, \frac {\partial \hat {\mathbf {u}}}{\partial t}, \frac {\partial^ {2} \hat {\mathbf {u}}}{\partial t ^ {2}}, \dots\right) = 0, \quad t \in \Upsilon
$$

where $\mathbf { x } \in \mathbb { R } ^ { d }$ is the spatial coordinate and t is the time; $\mathcal { F }$ denotes the residual of the PDE, containing the differential operators $( \mathrm { i . e . } \ \partial _ { \mathbf { x } } \hat { \mathbf { u } } , \partial _ { t } \hat { \mathbf { u } } , . . . ) ; \mu = [ \mu _ { 1 } , \mu _ { 2 } , . . . ]$ are the PDE parameters; $\hat { \mathbf { u } } ( \mathbf { x } , t )$ is the solution of the PDE with initial condition $\mathcal { C }$ and boundary condition B (which can be Dirichlet, Neumann or mixed); Ω, Γ and $\Upsilon$ represent the spatial domain resp. boundary. A special example considered in this paper is the Burgers equation (given in Eq. 12): $\partial _ { t } \hat { \mathbf { u } } + \hat { \mathbf { u } } \partial _ { \mathbf { x } } \hat { \mathbf { u } } - \nu \partial _ { \mathbf { x } } ^ { 2 } \hat { \mathbf { u } } = 0$ with PDE parameter $\mu$ as viscosity coefficient $\nu .$ . This paper is concerned with solving forward as well as inverse problems from different fields of application. For the forward problem, solutions of PDEs are to be inferred with fixed parameters $\mu ,$ while for the inverse problem setting, $\mu$ is unknown and has to be learned from observed data together with the PDE solution. 

Following the "vanilla" implementation of PINNs [48], a fully-connected feed-forward neural network (FCNN) $U ( \mathbf { x } , t ; \theta )$ is used to approximate the function $\hat { \mathbf { u } } ( \mathbf { x } , t )$ which solves the PDE. A FCNN consists of multiple hidden layers with trainable parameters (weights and biases; denoted by $\theta )$ and takes as inputs the space and time coordinates $\mathbf { \Psi } ( \mathbf { x } , t )$ , cf. fig. 1. The losses are then defined as follows: 

$$
\mathcal {L} _ {\Omega} = \frac {1}{| \hat {\Omega} |} \sum_ {\mathbf {x}, t \in \hat {\Omega}} \left\| \mathcal {F} \left(\mathbf {u}, \frac {\partial \mathbf {u}}{\partial t}, \frac {\partial \mathbf {u}}{\partial \mathbf {x}}, \frac {\partial^ {2} \mathbf {u}}{\partial \mathbf {x} ^ {2}}, \dots , \mu\right) \right\| _ {2} ^ {2}
$$

$$
\mathcal {L} _ {\Gamma_ {i}} = \frac {1}{| \hat {\Gamma} _ {i} |} \sum_ {\mathbf {x}, t \in \hat {\Gamma} _ {i}} \left\| \mathcal {B} _ {i} \left(\mathbf {u}, \frac {\partial \mathbf {u}}{\partial \mathbf {x}}, \frac {\partial^ {2} \mathbf {u}}{\partial \mathbf {x} ^ {2}}, \dots\right) \right\| _ {2} ^ {2} \tag {2}
$$

$$
\mathcal {L} _ {\Upsilon_ {i}} = \frac {1}{| \hat {\Upsilon} _ {i} |} \sum_ {\mathbf {x}, t \in \hat {\Upsilon} _ {i}} \left\| \mathcal {C} _ {i} \left(\mathbf {u}, \frac {\partial \mathbf {u}}{\partial t}, \frac {\partial^ {2} \mathbf {u}}{\partial t ^ {2}}, \dots\right) \right\| _ {2} ^ {2}
$$

$$
\mathcal {L} _ {\Psi} = \frac {1}{| \hat {\Psi} |} \sum_ {\mathbf {x}, t \in \hat {\Psi}} \| \mathbf {u} - d (\mathbf {x}, t) \| _ {2} ^ {2}
$$

where $\hat { \Omega }$ is a set of collocation points on the physical domain, $\hat { \Gamma } _ { i }$ for the boundary conditions $( \mathrm { B C } ) , \hat { \Upsilon } _ { i }$ for the initial conditions (IC) and $\check { \hat { \Psi } }$ represents a set of measurements (data); the function $d$ maps $\mathbf { \rho } ( \mathbf { x } , t ) $ to measurements at those coordinates; u is the output from the neural network $U ( \mathbf { x } , t ; \theta )$ . PINNs are generally trained using the L2-norm (mean squared error / MSE) on uniformly sampled collocation points defined as a data set $\{ \mathbf { x } _ { i } , t _ { i } \} _ { i = 1 } ^ { N }$ prior to training. Note that the number of points $N$ (denoted by | · | in Eq. 2) may vary for different loss terms. 

The objectives in $\operatorname { E q }$ . 2 are trained jointly and hence fall into the class of multi-objective optimisation (MOO) (cf. Eq. 4) 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-11/d96ef475-e7f5-4080-ac1a-f1d5c66d9f71/bbe2fad2606ecfa072ba48e855277ce8e08b2d2ac69ab0e2fb233bcef0777bd9.jpg)



Figure 1: Schematic of a Physics-Informed Neural Network (PINN): A fully-connected feed-forward neural network with space and time coordinates $\mathbf { \Psi } ( \mathbf { x } , t )$ as inputs, approximating a solution $\hat { \mathbf { u } } ( \mathbf { x } , t )$ . Derivatives of u w.r.t. inputs are computed by automatic differentiation (AD) and then incorporated into residuals of the governing equations as the loss function, which is composed of multiple terms weighted by different coefficients. Parameters of the FCNN θ and the unknown PDE parameters $\mu$ may be optimised simultaneously by minimising the loss function.


$$
\mathcal {L} (\theta , \mu) = (\mathcal {L} _ {\Omega}, \mathcal {L} _ {\Gamma_ {1}}, \dots , \mathcal {L} _ {\Gamma_ {n}}, \mathcal {L} _ {\Upsilon_ {1}}, \dots , \mathcal {L} _ {\Upsilon_ {m}}, \mathcal {L} _ {\Psi}) ^ {T} \tag {3}
$$

where the individual terms can be interpreted in the following way: 

• the first term ${ \mathcal { L } } _ { \Omega }$ penalises the residual of the governing equations (PDEs), included in both the forward and inverse problem. 

• the following n terms $\mathcal { L } _ { \Gamma }$ enforce the boundary conditions (BCs), included only in the forward problem. 

• the following m terms $\mathcal { L } _ { \Upsilon _ { \smash { \mathstrut } } }$ enforce the initial conditions (ICs), included only in the forward problem. 

• the last term ${ \mathcal { L } } _ { \Psi }$ makes the network approximate the measurements, included both in the forward (albeit not strictly necessary) and inverse problem. 

A common approach for handling MOO is through linear scalarisation described in more detail in the following Section IV. 

## IV. Multi-Objective Optimisation

Multi-objective optimisation (MOO) is concerned with simultaneously optimising a set of $k \mathbf { \Omega } > \mathbf { \Omega } 1$ , potentially conflicting objectives [5, 27]. 

$$
\mathcal {L} (\theta) = \left(\mathcal {L} _ {1} (\theta), \dots , \mathcal {L} _ {k} (\theta)\right) ^ {T} \tag {4}
$$

which can be turned into a single objective through linear scalarisation: 

$$
\mathcal {L} (\theta) = \sum_ {i = 1} ^ {k} \lambda_ {i} \mathcal {L} _ {i} (\theta), \quad \lambda_ {i} \in \mathbb {R} _ {> 0} \tag {5}
$$

Many problems in engineering, natural sciences, or economics can be formulated as multi-objective optimisations and generally require trade-offs to simultaneously satisfy all objectives to a certain degree [7]. The solution of MOO models is usually expressed as a set of Pareto optima, representing these optimal trade-offs between given criteria according to the following definitions [53]: 

Definition IV.1. A solution $\hat { \theta } \in \Omega$ Pareto dominates solution θ (denoted $\hat { \theta } \prec \theta )$ if and only if $\mathcal { L } _ { i } ( \hat { \theta } ) \leq \mathcal { L } _ { i } ( \theta )$ , ∀i ∈ $\{ 1 , \ldots , m \}$ and $\exists j \in \{ 1 , \ldots , m \}$ such that $\mathcal { L } _ { j } ( \hat { \theta } ) < \mathcal { L } _ { j } ( \theta )$ . 

Definition IV.2. A solution $\hat { \theta } \in \Omega$ is said to be Pareto optimal if $\forall \theta \in \Omega , { \widehat { \theta } } \preceq \theta$ . The set of all Pareto optimal points is called the Pareto set and the image of the Pareto set in the loss space is called the Pareto front. 

In theory, a Pareto optimal solution θ is independent of the scalarisation [52]. However, when using neural networks for MOO, the solution space becomes highly non-convex. Thus, although neural networks are universal function approximators [21], they are not guaranteed to find the globally optimal solution through gradient-based optimisation. Scaling the loss space therefore provides the option of guiding the gradients into having an a priori deemed desirable property. However, manually finding optimal $\lambda _ { \mathbf { i } }$ requires laborious grid search and becomes intractable as k gets large. Furthermore, one might want to let $\lambda _ { \mathbf { i } }$ evolve over time. This raises the need for an automated scheme to dynamically choose the scalings $\lambda _ { \mathbf { i } }$ . 

## V. Adaptive Loss Balancing Schemes

This section reviews different methods aiming at balancing the various terms within multi-objective optimisation. To this end, we compare the effectiveness of Learning Rate Annealing [65], proposed in the context of PINNs as well as two approaches originating from Computer Vision applications: GradNorm [8] and SoftAdapt [20]. This forms the basis for deriving and presenting our own loss balancing method as given in sec. VI. 

## A. Learning Rate Annealing

Wang et al. [65] conducted a study on gradients in PINNs and identified pathologies that explained some failure modes. One pathology is gradient stiffness in the boundary conditions caused by the imbalance amongst the different loss terms. As a remedy, it is proposed to adaptively scale the loss using gradient statistics, thus reducing the laborious tuning of these hyperparameters. 

$$
\hat {\lambda} _ {i} (t) = \frac {\max \{| \nabla_ {\theta} \mathcal {L} _ {\Omega} (t) | \}}{| \nabla_ {\theta} \mathcal {L} _ {\{\Gamma , \Upsilon \} _ {i}} (t) |}, i \in \{1, \dots , k \}
$$

$$
\lambda_ {i} (t) = \alpha \lambda_ {i} (t - 1) + (1 - \alpha) \hat {\lambda} _ {i} (t) \tag {6}
$$

$$
\theta^ {(t + 1)} = \theta^ {(t)} - \eta \nabla_ {\theta} \left(\mathcal {L} _ {\Omega} (t) + \sum_ {i = 1} ^ {k} \lambda_ {i} (t) \mathcal {L} _ {\{\Gamma , \Upsilon \} _ {i}} (t)\right)
$$

where $\overline { { | \nabla _ { \theta } \mathcal { L } _ { \Gamma _ { i } } ( t ) | } }$ is the mean of the gradient $\mathrm { w . r . t }$ . the parameters $\theta ;$ α is a hyperparameter with a value $\alpha = 0 . 9$ recommended by the authors. 

With this method, whenever the maximum value of $| \nabla _ { \theta } \mathcal { L } _ { \Omega } ( t ) |$ grows considerably larger than the average value in $| \nabla _ { \theta } \mathcal { L } _ { \mathrm { \{ { r , } \Upsilon \} } } ( t ) |$ |, the scalings $\hat { \lambda } _ { \bf i } ( t )$ correct for this discrepancy such that all gradients have similar magnitudes. Additionally, exponential decay is used in order to smoothen the balancing and avoid drastic changes of the loss space between optimisation steps. 

This procedure induces a few drawbacks. Its unboundedness potentially involves up- or down-scaling of terms by means of several orders of magnitude. The up-scaling in particular can cause problems similar to the effect of choosing a learning rate that is too large and therefore leads to repeatedly overshooting the objective. Furthermore, scaling all terms to have the same magnitude throughout training can incite the network to optimise for the "lowhanging fruit". A term, whose loss decreased considerably in the last optimisation step, will see its contribution to the total gradient scaled back up to the same magnitude to match the other terms. Therefore, the network might focus on the objectives that are easiest to optimise for. 

## B. GradNorm

Chen et al. [8] take a different approach and make the scalings λi trainable. The updates on these trainable scalings are chosen such that all terms improve at the same relative rate w.r.t. their initial loss and performed by a separate optimiser. A term that improved at a higher rate since the beginning of training compared to the other terms, gets a weaker scaling until all terms have made the same relative progress. Therefore, one could argue that they weakly enforce each optimisation step to Pareto dominate (cf. definition IV.1) its predecessor. The loss for updating the scalings within GradNorm is computed as follows: 

$$
\mathcal {L} (t; \lambda) = \sum_ {i = 1} ^ {k} \left| G _ {\theta} ^ {(i)} (t) - \overline {{G}} _ {\theta} (t) \times [ r _ {i} (t) ] ^ {\alpha} \right| _ {1} \tag {7}
$$

where $G _ { \theta } ^ { ( i ) } ( t ) ~ = ~ \| \nabla _ { \theta } \lambda _ { i } \mathcal { L } _ { i } ( t ) \| _ { 2 }$ is the $L _ { 2 }$ norm of the gradient w.r.t. the network parameters θ for the scaled $\begin{array} { r } { i \in \{ 1 , \ldots , \hat { k } \} ; \overline { { G } } _ { \theta } ( t ) = \frac { 1 } { k } \sum _ { i = 1 } ^ { k } G _ { \theta } ^ { ( i ) } ( t ) } \end{array}$ is the average of all gradient norms; $r _ { i } ( t ) = \dot { \mathcal { L } _ { i } } ( t ) / ( \mathcal { L } _ { i } ( 0 ) { \cdot } \overline { { \mathcal { L } } } ( t ) )$ defines the rate at which term i improved so far; α is a hyperparameter representing the strength of the restoring force which pulls tasks back to a common training rate. $\overline { { G } } _ { \theta } ( t ) \times [ r _ { i } ( t ) ] ^ { \alpha }$ $\bar { G } _ { \theta } ^ { ( i ) } ( t )$ should take on, so gradients must be prevented from flowing through this expression. The final loss for updating the networks parameters is then simply a linear scalarisation with the scalings that were previously updated: 

$$
\mathcal {L} (t; \theta) = \sum_ {i = 1} ^ {k} \lambda_ {i} (t) \mathcal {L} _ {i} (t) \tag {8}
$$

This algorithm is fairly evolved and, despite solving some of Learning Rate Annealing’s issues, it still requires a separate backward-pass for each task, which becomes prohibitively expensive as k gets large. Furthermore, it relies on two separate optimisation rounds at each step: one for adapting the scalings $\lambda _ { i }$ and another for updating the weights θ. By means of Eq. 4, GradNorm can thus be formulated as a scalarised MOO via: 

$$
\mathcal {L} (t) = \left(\mathcal {L} (t; \theta), \mathcal {L} (t; \lambda)\right) ^ {T} \tag {9}
$$

which in turn requires empirical hyperparameter tuning (learning rate, initialisation, etc.) to keep the system balanced - exactly the problem we are actually trying to solve through the use of adaptive loss balancing schemes. 

## C. SoftAdapt

Similar to GradNorm, SoftAdapt [20] leverages the ansatz of relative progress in order to balance the loss terms. However, the authors relax it by only considering the previous time-step $\mathcal { L } _ { i } ( t - 1 )$ and taking the difference between time steps instead of the division. The scalings are then normalised by using a softmax function: 

$$
\lambda_ {i} (t) = \frac {\exp \left(\mathcal {T} (\mathcal {L} _ {i} (t) - \mathcal {L} _ {i} (t - 1))\right)}{\sum_ {j = 1} ^ {k} \exp \left(\mathcal {T} (\mathcal {L} _ {j} (t) - \mathcal {L} _ {j} (t - 1))\right)}, i \in \{1, \dots , k \} \tag {10}
$$

where $\mathcal { L } _ { i } ^ { ( t ) }$ is the loss of term i at optimisation step t. 

SoftAdapt also differs from GradNorm in the sense that it does not require gradient statistics and thus eliminates the need of performing separate backward passes for each objective. Instead, it makes use of the fact that magnitudes in the gradients directly depend on the magnitudes of the terms in the loss function and therefore aims at achieving the balance solely through loss statistics. This is obviously true only if the same loss function is used for every objective (e.g. the $L _ { 2 }$ loss). However, this setting generalises to a vast majority of applications involving PINNs. 

## VI. Relative Loss Balancing with Random Lookback (ReLoBRaLo)

Drawing inspiration from existing balancing techniques as outlined in sec. ${ \mathrm { V } } ,$ we propose a novel method and implementation for balancing the multiple terms in the scalarised MOO loss function for training of PINNs upon: 

• SoftAdapt’s concept of operating on loss statistics as opposed to gradient statistics is employed. A computationally inexpensive softmax ensures the sum of scalings is bounded. 

• Inspired by GradNorm, the progress is calculated by dividing the loss at the current iteration $\mathcal { L } _ { i } ( t )$ by the loss at the previous iteration $\mathcal { L } _ { i } ( t - 1 )$ . 

• Similarly to Learning Rate Annealing, the scalings are updated using an exponential decay in order to utilise loss statistics from more than just one training step in the past. 

• In addition, a random lookback (called saudade $\rho )$ is introduced into the exponential decay, which decides whether to use the previous steps’ loss statistics to compute the scalings, or whether to look all the way $\mathcal { L } _ { i } ^ { ( 0 ) }$ 

$$
\lambda_ {i} ^ {b a l} (t, t ^ {\prime}) = m \cdot \frac {\exp \left(\frac {\mathcal {L} _ {i} (t)}{\mathcal {T L} _ {i} (t ^ {\prime})}\right)}{\sum_ {j = 1} ^ {m} \exp \left(\frac {\mathcal {L} _ {j} (t)}{\mathcal {T L} _ {j} (t ^ {\prime})}\right)}, i \in \{1, \ldots , m \}
$$

$$
\lambda_ {i} ^ {h i s t} (t) = \rho \lambda_ {i} (t - 1) + (1 - \rho) \lambda_ {i} ^ {b a l} (t, 0))
$$

$$
\lambda_ {i} (t) = \alpha \lambda_ {i} ^ {\text { hist }} + (1 - \alpha) \lambda_ {i} ^ {\text { bal }} (t, t - 1) \tag {11}
$$

where α is the exponential decay rate, $\rho$ is a Bernoulli random variable and $\mathbb { E } [ \rho ]$ should be chosen close to 1. The intermediate step $\lambda _ { i } ^ { b a l } ( t , t ^ { \prime } )$ calculates scalings based on the relative improvements of each term between time steps $t ^ { \prime }$ and t. The following step $\lambda _ { i } ^ { h i s t } ( t )$ defines, whether the scalings calculated in the previous time step $( \rho$ evaluates to 1) or the relative improvements since the beginning of training $( \rho$ evaluates to 0) should be carried forward. Note that this concept of randomly retaining or discarding the history of scalings is what we denote as "random lookbacks". Finally, the scaling $\lambda _ { i } ( t )$ for term i is obtained by means of an exponential decay, where α controls the weight given to past scalings versus the scalings calculated in the current time step. 

This method is an attempt at combining the best attributes of the aforementioned approaches into a new scheme for scalarised MOO objective functions. First and foremost, it still weakly enforces every training step to Pareto dominate its predecessor, which is an important property in physical applications. It also avoids using gradient statistics, making it considerably more efficient than Learning Rate Annealing and GradNorm. Furthermore, it reduces drastic changes in the loss space by using exponential decay and can easily be adapted to use more or fewer information of past optimisation steps by tuning the hyperparameter α. One can think of α as the model’s ability to remember the past, with a high alpha giving lots of weight to past loss statistics, while a lower alpha increases stochasticity. Setting $\alpha = 1$ results in each term’s relative progress being computed w.r.t. the initial loss $\mathcal { L } _ { i } ^ { ( 0 ) }$ i 

it causes the model to stop making progress as soon as one term reaches a local minimum. We chose values α between 0.9 and 0.999 and report the effects of varying this hyperparameter in sec. IX. 

Choosing the value of α also requires to make a trade-off: a high value means the model will remember potential deterioration of certain terms for longer and therefore leave a longer time frame in order to compensate them. However, it also induces a latency between a term starting to deteriorate and the scalings $\lambda _ { i }$ reacting accordingly. We therefore study the effect of introducing the saudade Bernoulli random variable $\rho$ that causes the model to occasionally look back until the start of training. $\mathbb { E } [ \rho ] = 0$ is maximum saudade as it always takes the loss value of the initial training step, while $\mathbb { E } [ \rho ] = 1$ corresponds to minimum saudade, taking only into account the last value from the history of the i-th scaling factor. Selecting $\mathbb { E } [ \rho ]$ somewhere between 0 and 1 allows to set a lower value for $\alpha ,$ thus making the model more flexible while still occasionally "reminding" it of the progress made since the start of training. Furthermore, the random lookback can give episodic new impulses and let the model escape local minima by changing the loss space, as well as inciting it to explore more of the parameter space. In case the impulse would turn out to have a negative effect on the accuracy, one can still choose to roll back and reset the network’s parameters θ to the previous state. 

The last hyperparameter is the so-called temperature $\tau$ Setting $\mathcal T \to \infty$ re-calibrates the softmax to output uniform values and thus all $\hat { \lambda } _ { i } ^ { ( t ) } = 1$ . On the other hand, $\mathcal { T }  0$ essentially turns the softmax into an argmax function, with the scaling $\hat { \lambda } _ { i } ^ { ( t ) } = k$ resulting for the term with the lowest relative progress and $\hat { \lambda } _ { i } ^ { ( t ) } = \bar { 0 }$ for all others. A pedagogical example with interpretation of expected behaviours and how to draw conclusions from the histories of the scalings is given for Burgers’ equation in sec. VIII-A. 

Note that the network should be prevented from optimising this expression. This can be achieved by stopping the gradients from flowing through the calculation of the scalings. Also, depending on the problem at hand, $\exp ( \mathcal { L } _ { i } ( t ) / ( \mathcal { T } \mathcal { L } _ { i } ( t ^ { \prime } ) ) )$ could evaluate to a very large number, thus leading to overflows. This issue can be preemptively tackled by subtracting a large number $\left( \mathrm { e . g . ~ } 1 0 ^ { - 9 } \right)$ from the input to the softmax. 

## VII. Hyperparameter Tuning and Meta Learning

This paper uses grid search in combination with Bayesian Optimisation (BO) [36, 43, 59] for hyperparameter tuning. This study uses hyperparameters for defining the NN architecture $( d _ { K }$ hidden layers and $w _ { K }$ neurons per layer) and training settings (learning rate $l _ { r } ,$ exponential decay rate α and saudade $\rho )$ . Tab. I contains the ranges and distributions for the hyperparameters. Bayesian Optimisation reduces the empiricism of selecting the PINNs hyperparameters to learn an optimal NN structure. First, 20 random points in the hyperparameter space are sampled and evaluated. The model’s performance at those points serves as evidence for fitting prior Gaussian Processes in order to estimate the unknown loss function w.r.t. the hyperparameters. Using Expected Improvement (EI) [41], further 80 points are then sampled and evaluated to refine the prior. This procedure provides an educated guess as to which are the optimal hyperparameters for the task at hand. Finally, we fine-tune the results by performing fine-grained grid search around the hyperparameters returned by Bayesian Optimisation (BO). 

<table><tr><td>Hyperparameter</td><td>Range</td><td>Log-scaling</td></tr><tr><td>Learning Rate <eq>l_r</eq></td><td><eq>[10^{-6}, 10^{-2}]</eq></td><td>yes</td></tr><tr><td>Layers <eq>d_K</eq></td><td>[2, 4]</td><td>no</td></tr><tr><td>Neurons per Layer <eq>w_K</eq></td><td>[32, 512]</td><td>no</td></tr><tr><td>Exponential Decay Rate <eq>\alpha</eq></td><td>[0, 1]</td><td>no</td></tr><tr><td>Temperature <eq>\mathcal{T} = 10^t</eq></td><td><eq>[10^{-6}, 10^{2}]</eq></td><td>yes</td></tr><tr><td>Expected Saudade <eq>\mathbb{E}[\rho]</eq></td><td>[0, 1]</td><td>no</td></tr><tr><td>Activation function <eq>\sigma</eq></td><td><eq>\{tanh, sigmoid\}</eq></td><td>no</td></tr></table>

Table I: Hyperparameters for architecture and training settings together with ranges as used for Bayesian Optimisation. 

Within this study, the exact same Bayesian Optimisation configuration was used for all examples presented in sec. VIII, hence it is sufficient to only display tab. I. Respective results of the Grid Search and Bayesian Optimisation can also be found in sec. VIII. 

## VIII. Results

We evaluate the different balancing schemes on three problems (Burgers equation, Kirchhoff plate bending and Helmholtz equation) originating from physics-informed deep learning, where the objective function consists of various terms of potentially considerably different magnitudes and compare their performances, as well as their computational efficiency. Training was done on networks of varying depth and width (acc. to tab. I) and limited to $1 0 ^ { 5 }$ steps of gradient descent (GD) using the Adam optimiser [31]. Additionally, we reduced the learning rate by a multiplicative factor of 0.1 whenever the optimisation stopped making progress for over 3’000 optimisation steps and finally used early stopping in case of 9’000 steps without improvement. When addressing the inverse problem, i.e. approximating a set of measurements while subjecting the network to PDE constraints for finding unknown PDE parameters $\mu ,$ we further investigated the payoff of using two separate optimisers: one for updating network weights θ, and a separate one for updating PDE parameters $\mu .$ Further details on hyperparameter tuning and meta learning is given in sec. IX. 

## A. Burgers’ Equation

Burgers’ equation is a one-dimensional PDE describing the main properties of the Navier-Stokes equations [44] used i.a. to model shock waves, gas dynamics, or traffic flow [3]. Using Dirichlet boundary conditions, the PDE takes the following form: 

$$
\begin{array}{l} \frac {\partial u}{\partial t} + u \frac {\partial u}{\partial x} - \nu \frac {\partial^ {2} u}{\partial x ^ {2}} = 0, \quad x \in [ - 1, 1 ], \quad t \in [ 0, 1 ] \\ u (0, x) = - \sin (\pi x) \end{array} \tag {12}
$$

$$
u (t, - 1) = u (t, 1) = 0
$$

At first, we investigate the solution of the forward problem, where we set the PDE parameter $\textstyle \mu : = \nu = { \frac { 1 } { 1 0 0 \pi } }$ In order to find the latent function $\hat { \mathbf { u } } ( \mathbf { x } , t )$ , we can parameterise it with a neural network $U ( x , t ; \theta )$ and turn the set of equations into a linear scalarised objective (cf. Eq. 5) of Mean Squared Errors (MSE). This loss function will weakly enforce the network to approximate the PDE solution uˆ(x, t). 

$$
\begin{array}{l} \mathcal {L} _ {\Omega} = \frac {1}{| \hat {\Omega} |} \sum_ {(x, t) \in \hat {\Omega}} \left\| \frac {\partial U}{\partial t} + U \frac {\partial U}{\partial x} - \nu \frac {\partial^ {2} U}{\partial x ^ {2}} \right\| _ {2} ^ {2} \\ \mathcal {L} _ {\Gamma_ {1}} = \frac {1}{| \hat {\Gamma} _ {1} |} \sum_ {t \in \hat {\Gamma} _ {1}} \| U (- 1, t; \theta) \| _ {2} ^ {2} \tag {13} \\ \mathcal {L} _ {\Gamma_ {2}} = \frac {1}{| \hat {\Gamma} _ {2} |} \sum_ {t \in \hat {\Gamma} _ {2}} \| U (1, t; \theta) \| _ {2} ^ {2} \\ \mathcal {L} _ {\Upsilon} = \frac {1}{| \hat {\Upsilon} |} \sum_ {x \in \hat {\Upsilon}} \| U (x, 0; \theta) + \sin (\pi x) \| _ {2} ^ {2} \\ \end{array}
$$

For the forward problem, PINNs training induces the following loss function employed during training: 

$$
\mathcal {L} = \lambda_ {0} \mathcal {L} _ {\Omega} + \lambda_ {1} \mathcal {L} _ {\Gamma_ {1}} + \lambda_ {2} \mathcal {L} _ {\Gamma_ {2}} + \lambda_ {3} \mathcal {L} _ {\Upsilon} \tag {14}
$$

After a successful convergence of the PINNs training using ReLoBRaLo, we obtain the results displayed in fig. 2(b), whereas the final algorithm settings are reported in tab. VIII. As there is no analytical solution available for the Burgers equation, we compared the results to a reference solution calculated using the finite element method (FEM), displayed in fig. 2(a). A plot of the squared difference in u as given by the FEM and PINNs is shown in fig. 2(c) and delivers a relative max error of below 5%. 

However, Burgers’ equation can also be turned into an inverse problem by regarding the PDE parameter ν as an unknown to be estimated from a set of observations (i.e. data) over the spatial and temporal domain. In this setting, the PINNs induced loss function to be deployed reads: 

$$
\mathcal {L} = \lambda_ {0} \frac {1}{| \hat {\Omega} |} \sum_ {(x, t) \in \hat {\Omega}} \left\| \frac {\partial U}{\partial t} + U \frac {\partial U}{\partial x} - \nu \frac {\partial^ {2} U}{\partial x ^ {2}} \right\| _ {2} ^ {2} \tag {15}
$$

$$
+ \lambda_ {1} \frac {1}{| \hat {\Psi} |} \sum_ {(x, t) \in \hat {\Psi}} \| U - u \| _ {2} ^ {2}
$$

Similar to the network’s weights and biases, the additional trainable PDE variable µ (here viscosity ν) is now also updated through gradient descent: 

$$
\theta^ {(t)} = \theta^ {(t - 1)} - \eta \nabla_ {\theta} \mathcal {L} \tag {16}
$$

$$
\mu^ {(t)} = \mu^ {(t - 1)} - \eta \nabla_ {\mu} \mathcal {L}
$$

Measurement data Ψ for the inverse problem setting were obtained from our reference solution computed using the FEM without addition of noise. At every iteration, we sample from the available data in order to generate a batch of collocation points. 


(a) FEM-Result


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-11/d96ef475-e7f5-4080-ac1a-f1d5c66d9f71/69ae10baee59d28885452d5c45c58fb26d6615c0adb8366f34fa4ea601b28b6c.jpg)



(b) PINN-Result


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-11/d96ef475-e7f5-4080-ac1a-f1d5c66d9f71/a44d81ca1c103adf5a6bb7409ed8b5bdac65eccae585cf07bc06c455c42049cf.jpg)



(c) Squared Error


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-11/d96ef475-e7f5-4080-ac1a-f1d5c66d9f71/6f621cc79af4e13adbd8ffd7fc59e2485569a4292e81fbcfd3551f7715ab3a00.jpg)



Figure 2: Burgers’ equation problem: (a) FEM reference solution, (b) PINNs results predicted with a fully-connected network consisting of two layers and 128 nodes each, and (c) squared error.



(i) $L _ { 2 }$ Convergence


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-11/d96ef475-e7f5-4080-ac1a-f1d5c66d9f71/7380323d27e860e933d9163cc37023a9d878e98a343272601b9053c2b8028791.jpg)



(ii) ReLoBRaLo Scaling


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-11/d96ef475-e7f5-4080-ac1a-f1d5c66d9f71/cf62cd34e6914eead816043eafed28b206e27f3ddcef875807dae60e6a9a7ce2.jpg)



(a) α = 0.9, T = 0.1, E[ρ] = 1


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-11/d96ef475-e7f5-4080-ac1a-f1d5c66d9f71/edaa47c0155ec72ccba19689fdc847d264b921473edabfde71c406b67b8f51d3.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-11/d96ef475-e7f5-4080-ac1a-f1d5c66d9f71/72b842135e6ad62998c7488eb7d2bb707a9f49f14be3aed47e86579e14532713.jpg)



(b) α = 0.999, T = 0.1, E[ρ] = 1


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-11/d96ef475-e7f5-4080-ac1a-f1d5c66d9f71/72b9ef3bbb58c8b8659a97f0f8b252994c86b5abbcac280318bdbb30ec3ef1e8.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-11/d96ef475-e7f5-4080-ac1a-f1d5c66d9f71/0c45feb3cb809bc2f76ecfbb63a00852bd774411bb04dbaca703ca86407f836a.jpg)



(c) α = 0.999, T = 1, E[ρ] = 1


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-11/d96ef475-e7f5-4080-ac1a-f1d5c66d9f71/fd8a5c31de9ef0995ad295ada2f25c17c651c5aefa720dd811063c534fd47274.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-11/d96ef475-e7f5-4080-ac1a-f1d5c66d9f71/d09bf0ee593c16c2f09bc3d8950058155208d62e2e4feb9ef0f52bbfa2a2d3eb.jpg)



(d) α = 0.999, T = 1, E[ρ] = 0.999



Figure 3: Median of the log $L _ { 2 }$ loss over multiple training runs $\mathrm { ( i / l e f t ) }$ of Burgers’ equation and the mean and variance of the corresponding scaling factors $\lambda \ \mathrm { ( i i / r i g h t ) }$ computed with ReLoBRaLo.


Fig. 3 shows the scaling factors $\lambda _ { i }$ of our ReLoBRaLo method with varying hyperparameters for Burgers’ equation. As can be expected, a larger value for α leads to a smoother curve because past loss statistics are dragged on longer, therefore countering the stochasticity that might arise at every optimisation step. On the other hand, the temperature $\tau$ influences the magnitude of the scalings. Another general tendency in the plots of the scaling factors is the fact that the relatively lower loss contributions (here: BC 1 and BC 2) correspond to higher scaling values (potentially greater than 1), while the larger loss contributions (here: PDE and IC) correspond to lower scaling values (potentially less than 1). Note that, whenever $" \log "$ is used in this and all subsequent figures, we refer to the natural logarithm of that quantity. 

The relatively small variances across training runs with $\mathbb { E } [ \rho ] = 0$ suggest that the optimisation progress follows similar patterns, even when varying depth and width of the network. Therefore, these values can provide valuable insight into the training and help to identify possibilities of improving the model. E.g. the fact that the scaling for the governing equation has the largest value after $5 0 { , } 0 0 0$ epochs indicates that it was the first term to stop making progress. We will see in the following sections that the opposite holds true for Helmholtz’ and Kirchhoff’s equations, where the boundary conditions have more difficulties making progress (cf. fig. 6). This knowledge can help taking informed decisions to improve the framework, e.g. by adapting the activation functions, the loss function or the model’s architecture accordingly. 

Tab. II summarises the performances of the different balancing techniques against a baseline for the forward and inverse problem setting, where we manually chose the optimal scalings $\lambda _ { i }$ through grid search. As can be observed, the adaptive scaling techniques perform similarly well to the baseline, with Learning Rate Annealing and ReLoBRaLo reaching a considerably lower validation error. The results show that either one of these methods greatly reduces the amount of work required for hyperparameter search, while still achieving great results with high probability. 

Besides accuracy, computational efficiency is another important metric for evaluating adaptive loss balancing methods. By designing our ReLoBRaLo method such that it requires only one backward pass, its computational overhead can be expected to be relatively small compared to GradNorm and Learning Rate Annealing, which both utilise gradient statistics and hence separate backward passes for each term. Indeed, tab. III shows that Burgers’ equation with its four terms in the loss function can be solved by ReLoBRaLo about 40% faster than Learning Rate Annealing and 70% faster than GradNorm and thus adds to efficiency and sustainability of PINNs training. Note that the reported values in tab. III stem from tasks where the balancing operation was performed at every optimisation step. Both GradNorm and Learning Rate Annealing can be made more efficient by updating the scaling terms once every arbitrary number of iterations. However, this introduces a trade-off between flexibility and efficiency and therefore an additional, very sensitive hyperparameter with a high impact on the method’s accuracy and efficiency. On the other hand, ReLoBRaLo adapts its scalings at every iteration and very low computational cost. 

<table><tr><td>Burgers</td><td></td><td>Baseline</td><td>GradNorm</td><td>LR anneal.</td><td>SoftAdapt</td><td>ReLoBRaLo</td></tr><tr><td rowspan="3">Forward</td><td>train f</td><td><eq>5.5 \cdot 10^{-4}</eq></td><td><eq>6.6 \cdot 10^{-4}</eq></td><td><eq>9.9 \cdot 10^{-4}</eq></td><td><eq>2.0 \cdot 10^{-4}</eq></td><td><eq>5.6 \cdot 10^{-5}</eq></td></tr><tr><td>val u</td><td><eq>1.2 \cdot 10^{-3}</eq></td><td><eq>2.0 \cdot 10^{-3}</eq></td><td><eq>1.6 \cdot 10^{-4}</eq></td><td><eq>8.1 \cdot 10^{-4}</eq></td><td><eq>1.4 \cdot 10^{-4}</eq></td></tr><tr><td>std val u</td><td><eq>5.7 \cdot 10^{-4}</eq></td><td><eq>2.1 \cdot 10^{-3}</eq></td><td><eq>2.3 \cdot 10^{-4}</eq></td><td><eq>9.5 \cdot 10^{-3}</eq></td><td><eq>6.8 \cdot 10^{-4}</eq></td></tr><tr><td>Inverse</td><td>val μ</td><td><eq>1.9 \cdot 10^{-10}</eq></td><td><eq>6.8 \cdot 10^{-5}</eq></td><td><eq>2.5 \cdot 10^{-11}</eq></td><td><eq>1.1 \cdot 10^{-7}</eq></td><td><eq>2.2 \cdot 10^{-10}</eq></td></tr><tr><td><eq>\nu = \frac{1}{100\pi}</eq></td><td>std μ</td><td><eq>1.2 \cdot 10^{-9}</eq></td><td><eq>5.1 \cdot 10^{-5}</eq></td><td><eq>3.4 \cdot 10^{-11}</eq></td><td><eq>5.7 \cdot 10^{-7}</eq></td><td><eq>2.1 \cdot 10^{-10}</eq></td></tr></table>


Table II: Comparison of the median $L _ { 2 }$ training and validation loss on Burgers’ equation against a baseline of manually chosen scalings. The reported values are the median over four independent runs with identical settings. Additionally, we report the standard deviation over the runs of the best performing model on the validation loss.



Convergence on inverse Burgers’ problem


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-11/d96ef475-e7f5-4080-ac1a-f1d5c66d9f71/06a84eb5535b95df01ca281208fe7318f99e6276c5ada1e71493ccddb1514c7a.jpg)



Figure 4: Approximation of the true PDE parameter value ν (dashed line) for the inverse problem setting of Burgers’ equation. Reported values are the mean (solid line) and standard deviation (shaded area) of four independent computational runs.


<table><tr><td></td><td>GradNorm</td><td>LR ann.</td><td>SoftAdapt</td><td>ReLoBRaLo</td></tr><tr><td><eq>\Delta T_{co}</eq> [s]</td><td>+10.6</td><td>+3.5</td><td>+0.4</td><td>+0.6</td></tr></table>

Table III: Median computational overhead $\Delta T _ { c o }$ (in s) per 1’000 optimisation steps compared to using no balancing scheme (3.7s) on Burgers’ equation. 

It is noteworthy that, while the forward problem induced a loss function consisting of four terms, the inverse problem requires only two terms (Eq. 15). Hence, selecting the scalings manually is significantly less time-consuming than it is for the forward problem. Consequently, tab. II shows that the baseline was harder to outperform, with Learning Rate Annealing (LR Annealing) and ReLoBRaLo being the only methods yielding better results. It is worth noting however that Learning Rate Annealing approximates the true value of ν significantly faster than ReLoBRaLo and is therefore the optimal choice for this particular problem setting, cf. fig. 4. Further conclusions and comparisons across different loss balancing methods are made in sec. IX. 

## B. Kirchhoff Plate Bending Equation

The Kirchhoff–Love theory of plates arose from civil and mechanical engineering and consists of a two-dimensional mathematical model used to determine stresses and deformations in thin plates subjected to forces and moments [1]. The Kirchhoff plate bending problem assumes that a midsurface plane can be used to represent a three-dimensional plate in two-dimensional form and together with a linear elastic material a fourth-order PDE can be derived to describe its mechanical behaviour: 

$$
\nabla^ {4} u (x, y) - \frac {p (x , y)}{D} = 0, \quad (x, y) \in \mathbb {R} _ {> 0} ^ {2} \tag {17}
$$

$$
D = \frac {E h ^ {3}}{1 2 (1 - \nu^ {2})}
$$

where $p ( x , y )$ is the load acting on the plate at coordinates $( x , y )$ ; D is the plate’s flexural stiffness computed with Young’s modulus E, the plate’s thickness h and Poisson’s ratio ν. The Kirchhoff plate bending problem poses several severe problems to FEM solutions [1], yet analytical solutions can be inferred e.g. using Fourier series for special cases such as an applied sinusoidal load: 

$$
\begin{array}{l} p (x, y) = p _ {0} \sin \left(\frac {x \pi}{a}\right) \sin \left(\frac {y \pi}{b}\right) \tag {18} \\ u (x, y) = \frac {p _ {0}}{\pi^ {4} D (\frac {1}{a ^ {2}} + \frac {1}{b ^ {2}}) ^ {2}} \sin \left(\frac {x \pi}{a}\right) \sin \left(\frac {y \pi}{b}\right) \\ \end{array}
$$

In this paper we consider a concrete plate of width $a =$ 10 m, length $b \ = \ 1 0 \mathrm { m } .$ , base load $p _ { 0 } ~ = ~ 0 . 0 1 5 \mathrm { M N m ^ { - 2 } }$ , Young’s modulus $E = 3 0 . 0 0 0 \mathrm { M N m ^ { - 2 } }$ , plate height $h =$ 0.2 m and Poisson’s ratio of $\nu = 0 . 2$ , as well as simply supported edge boundary conditions as it arises in typical civil engineering structures such as slabs [10]. We hence consider the following boundary conditions (BC): 


(a) Analytical Result [m]



(b) PINN-Result [m]



(c) Squared Error -m2


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-11/d96ef475-e7f5-4080-ac1a-f1d5c66d9f71/a399fab679e541e4812ac7ec034030ed9cdeb6478f877dceb07145913f977417.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-11/d96ef475-e7f5-4080-ac1a-f1d5c66d9f71/ca5a7cd4f94ce89c241adbf47ed4f0392091c86f65314807b67e9874fb425f18.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-11/d96ef475-e7f5-4080-ac1a-f1d5c66d9f71/f002ff37dd6d2203fc872f43a23aaf4e48ca55e1f597bab3ccad1371b2416e94.jpg)



Figure 5: Kirchhoff plate bending problem: (a) analytical reference solution [m], (b) PINNs results [m] predicted with a fully-connected network consisting of three layers and 128 nodes each, and (c) squared error $\left[ m ^ { 2 } \right]$ .


$$
\begin{array}{l} u (0, y) = u (a, y) = u (x, 0) = u (x, b) = 0 \\ \text {m} (0, x) = \text {m} (x, y) = \text {m} (x, 0) = \text {m} (x, b) = 0 \end{array} \tag {19}
$$

$$
m _ {x} (0, y) = m _ {x} (a, y) = m _ {y} (x, 0) = m _ {y} (x, b) = 0
$$

where $m _ { x }$ and $m _ { y }$ are bending moments computed as follows: 

$$
m _ {x} (x, y) = - D \left(\partial_ {x} ^ {2} u + \nu \partial_ {y} ^ {2} u\right)
$$

$$
m _ {y} (x, y) = - D \left(\nu \partial_ {x} ^ {2} u + \partial_ {y} ^ {2} u\right) \tag {20}
$$

$$
m _ {x y} (x, y) = - D (1 - \nu) \partial_ {x y} ^ {2} u
$$

In total, we obtain 8 boundary conditions and therefore 9 terms in the PINNs loss function, making this a challenging task for balancing the contributions of the various objectives: 

$$
\begin{array}{l} \mathcal {L} ^ {(t)} = \frac {\lambda_ {0}}{| \hat {\Omega} |} \sum_ {x, y \in \hat {\Omega}} \left\| \nabla^ {4} U (x, y; \theta) - \frac {p}{D} \right\| _ {2} ^ {2} \\ + \sum_ {i = 1} ^ {4} \frac {\lambda_ {i}}{| \hat {\Gamma} _ {i} |} \sum_ {x, y \in \hat {\Gamma} _ {i}} \| U (x, y; \theta) \| _ {2} ^ {2} \\ + \sum_ {i = 5} ^ {6} \frac {\lambda_ {i}}{| \hat {\Gamma} _ {i} |} \sum_ {x, y \in \hat {\Gamma} _ {i}} \| m _ {u x} (x, y) \| _ {2} ^ {2} \tag {21} \\ + \sum_ {i = 7} ^ {8} \frac {\lambda_ {i}}{| \hat {\Gamma} _ {i} |} \sum_ {x, y \in \hat {\Gamma} _ {i}} \| m _ {u y} (x, y) \| _ {2} ^ {2} \\ \end{array}
$$

After the successful convergence of the PINNs training, we obtain the results displayed in fig. 5(b) and compare it to the analytically available solution displayed in fig. 5(a). A plot of the squared difference in u as given by the analytical and PINNs results is shown in fig. 5(c) and delivers a negligible maximum error. The final algorithm settings are reported in tab. VIII. 

Fig. 6 shows an example of ReLoBRaLo’s training progress on Kirchhoff’s equation. In this particular example, one can notice the larger variance of scaling values towards the end of training. Also, the scalings did not converge towards the value 1, thus suggesting that the training finished without all terms having stopped making progress, i.e. the scalings for the boundary conditions on the moments (yellow) were increasing at the end of training, while the boundary conditions on the displacements (red) were decreasing. This gives a strong indication as to where the model’s limitations lie. In this case, additional attention should be paid to the moments, e.g. by selecting an activation function which is better behaved in the second derivative than tanh. Note how ReLoBRaLo in combination with early stopping weakly imposes Pareto optimal updates, as it gradually increases the weights of underperforming terms and eventually leads to a stop of the training process due to a lack of global progress. This is an important property in the context of PINNs, because optimising only a subset of terms in the loss can lead to unsatisfactory solutions from a physical perspective. 


(i) $L _ { 2 }$ Convergence


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-11/d96ef475-e7f5-4080-ac1a-f1d5c66d9f71/253d365a3c6dbbd687e41de0142a70050e9c7ac649bef566405514a8cb100ac1.jpg)



(ii) ReLoBRaLo Scaling


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-11/d96ef475-e7f5-4080-ac1a-f1d5c66d9f71/b5ac88236327fa6302ee03f0aed495ba33ebc58d4bb88103cb3193b0d52adc16.jpg)



Figure 6: Median of the log $L _ { 2 }$ loss over multiple training runs (a) and the mean and variance of the corresponding scaling factors λ (b) computed with ReLoBRaLo on Kirchhoff’s equation with $\alpha = 0 . 9 9 9$ , $\mathcal { T } = 0 . 1$ , $\mathbb { E } [ \rho ] = 1$ . For the sake of readability, the boundary conditions 1 − 4 and 5 − 8 were aggregated by taking the mean value.


Concerning performance, ReLoBRaLo outperforms the baseline and other algorithms by almost an order of magnitude in accuracy, while also yielding a very small standard deviation and hence being very consistent across training runs (cf. tab. IV). The results show its effectiveness, even on Kirchhoff’s challenging problem with a total of 9 terms (cf. Eq. 21). Furthermore, the execution times in tab V underline the efficiency benefit (up to sixfold speedup) of balancing the loss without gradient statistics, as separate backwards passes for each term become increasingly computationally expensive as the number of terms in the loss function grows. Further conclusions and comparisons across different loss balancing methods are made in sec. IX. 

For the inverse Kirchhoff problem setting, we select the PDE parameter $\mu : = D$ (i.e. flexural stiffness) to be learned for given data, which we obtained by sampling from the analytically known solution. More specifically, we initialised $\mu \ = \ 0 . 5$ and tasked the network with approximating $\begin{array} { l } { \displaystyle { D \ \mathrm { ~ = ~ } \ 2 0 . 8 \overline { { 3 } } } } \end{array}$ . Given the large disparity between the initialisation and the target, we empirically found the use of two separate optimisers beneficial in this case, where one optimiser is used for updating the network’s parameters θ and a different one for updating the PDE parameter $\mu .$ Differently from Burgers’ equation, ReLoBRaLo also sets a new benchmark in Kirchhoff’s inverse problem, both in accuracy as well as convergence speed, cf. fig. 7 and tab. V. 

<table><tr><td>Kirchhoff</td><td></td><td>Baseline</td><td>GradNorm</td><td>LR anneal.</td><td>SoftAdapt</td><td>ReLoBRaLo</td></tr><tr><td rowspan="3">Forward</td><td>train <eq>f</eq></td><td><eq>1.2\cdot10^{-7}</eq></td><td><eq>5.3\cdot10^{-7}</eq></td><td><eq>9.1\cdot10^{-9}</eq></td><td><eq>1.8\cdot10^{-8}</eq></td><td><eq>6.0\cdot10^{-9}</eq></td></tr><tr><td>val <eq>u</eq></td><td><eq>1.3\cdot10^{-8}</eq></td><td><eq>1.7\cdot10^{-8}</eq></td><td><eq>2.7\cdot10^{-9}</eq></td><td><eq>2.5\cdot10^{-9}</eq></td><td><eq>4.0\cdot10^{-10}</eq></td></tr><tr><td>std val <eq>u</eq></td><td><eq>3.9\cdot10^{-8}</eq></td><td><eq>2.2\cdot10^{-7}</eq></td><td><eq>1.0\cdot10^{-6}</eq></td><td><eq>1.9\cdot10^{-9}</eq></td><td><eq>7.7\cdot10^{-10}</eq></td></tr><tr><td>Inverse</td><td>val <eq>\mu</eq></td><td>2.1</td><td>3.6</td><td>6.0</td><td>9.5</td><td><eq>3.2\cdot10^{-2}</eq></td></tr><tr><td><eq>D=20.8\overline{3}</eq></td><td>std <eq>\mu</eq></td><td>1.6</td><td>4.7</td><td>0.8</td><td>4.9</td><td><eq>2.9\cdot10^{-2}</eq></td></tr></table>


Table IV: Comparison of the median $L _ { 2 }$ training and validation loss on Kirchhoff’s equation against a baseline of manually chosen scalings. The reported values are the median over four independent runs with identical settings. Additionally, we report the standard deviation over the runs of the best performing model on the validation loss.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-11/d96ef475-e7f5-4080-ac1a-f1d5c66d9f71/8b588e5a381cf360c9345008932a6a5f5c02e70926ffd5160218858ac5cf93a3.jpg)



Figure 7: Approximation of the true PDE parameter value D (dashed line) for the inverse problem setting of Kirchhoff plate bending. Reported values are the mean (solid line) and standard deviation (shaded area) of four independent runs.


<table><tr><td></td><td>GradNorm</td><td>LR ann.</td><td>SoftAdapt</td><td>ReLoBRaLo</td></tr><tr><td><eq>\Delta T_{co}</eq> [s]</td><td>128.6</td><td>139.7</td><td>20.2</td><td>22.5</td></tr></table>


Table V: Median computational overhead $\Delta T _ { c o }$ (in s) per 1’000 optimisation steps compared to using no balancing scheme (17.3s) on Kirchhoff’s equation.


## C. Helmholtz equation

The Helmholtz equation represents a time-independent form of the wave equation and arises in many physical and engineering problems such as acoustics and electromagnetism [60]. The equation has the form: 

$$
\Delta u (x, y) + k ^ {2} u (x, y) = f (x, y), \quad x, y \in [ - 1, 1 ] ^ {2} \tag {22}
$$

where k is the wave number. This represents a common problem to benchmark PINNs and possesses an analytical solution in combination with Dirichlet boundaries: 

$$
\begin{array}{l} f (x, y) = \left(- \pi^ {2} - (4 \pi) ^ {2} + k ^ {2}\right) \sin (\pi x) \sin (4 \pi y) \\ u (x, y) = \sin (\pi x) \sin (4 \pi y) \tag {23} \\ \end{array}
$$

$$
u (- 1, y) = u (1, y) = u (x, - 1) = u (x, 1) = 0
$$

Both, the $x _ { 1 }$ and $x _ { 2 }$ input variables, are bounded below by -1 and bounded above by 1. Therefore, the boundary conditions add four terms to the loss function of the forward problem, resulting in a 5-term total physics-informed loss: 

$$
\begin{array}{l} \mathcal {L} ^ {(t)} = \frac {\lambda_ {0}}{| \hat {\Omega} |} \sum_ {x, y \in \hat {\Omega}} \left\| \Delta U (x, y; \theta) + k ^ {2} U (x, y; \theta) - f (x, y) \right\| _ {2} ^ {2} \\ + \sum_ {i = 1} ^ {4} \frac {\lambda_ {i}}{\left| \hat {\Gamma} _ {i} \right|} \sum_ {x, y \in \hat {\Gamma} _ {i}} \| U (x, y; \theta) \| _ {2} ^ {2} \tag {24} \\ \end{array}
$$

where $U ( x , y ; \theta )$ is the parameterisation of the latent function $\hat { \mathbf { u } } ( x , y )$ using a neural network with parameters θ. 

After a successful convergence of the PINNs training, we obtain the results displayed in fig. 8(b) and compare it to the analytically available solution displayed in fig. 8(a). A plot of the squared difference in u as given by the analytical and PINNs results is shown in fig. 8(c) and delivers a negligible max error. The final algorithm settings are reported in tab. VIII. 

The Helmholtz equation reveals a limitation of our basic loss balancing approach and motivates the introduction of the random lookback. GradNorm and Learning Rate Annealing both achieve impressive results and substantially outperform the baseline as well as ReLoBRaLo with $\mathbb { E } [ \rho ] ~ = ~ 1$ in terms of L2 accuracy for the BC terms, cf. fig. 9. This is likely due to the considerable initial difference in magnitudes between the governing equation and the boundary conditions. Furthermore, the high values of $\alpha ,$ necessary for "remembering" the deteriorations longer, induce a latency between the increase of a term’s loss until the scaling λ reacts accordingly (cf. fig. 9(d)). On the other hand, GradNorm and Learning Rate Annealing do not succeed in decreasing the L2 error as much as ReLoBRaLo for the governing equation term. Fig. 9 shows that both GradNorm and Learning Rate Annealing focus on improving the boundary conditions right from the beginning of training, whereas ReLoBRaLo with $\mathbb { E } [ \rho ] = 1$ counters the initial deterioration, but eventually "forgets" 


(a) Analytical Result


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-11/d96ef475-e7f5-4080-ac1a-f1d5c66d9f71/76799599590156d9544d2a6a0deec3231200c3262cf3b469aede35b999789ca4.jpg)



(b) PINN-Result


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-11/d96ef475-e7f5-4080-ac1a-f1d5c66d9f71/1196cfa01cd2b98da3ab5435744833dfcbe4152052d38b0ca45a66dc5ee832c4.jpg)



(c) Squared Error


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-11/d96ef475-e7f5-4080-ac1a-f1d5c66d9f71/48daacd2ce103f905716913d6bd94824ff1da0a0259035cb4be64afd4973bc32.jpg)



Figure 8: Helmholtz’s problem: (a) analytical reference solution, (b) PINNs results predicted with a fully-connected network consisting of two layers and 128 nodes each, and (c) squared error.



(a) GradNorm Result


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-11/d96ef475-e7f5-4080-ac1a-f1d5c66d9f71/3d8c62b4abd3bdc4b5d946ebf82c4e4d278c5b553f6e9fd9218b96bc830aa070.jpg)



(b) LR Annealing Result


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-11/d96ef475-e7f5-4080-ac1a-f1d5c66d9f71/bee9b2b2a7ac7c41d6f9d29a011654061bd0442dcd65a3ce5e8e319f7ac8ce4c.jpg)



(c) ReLoBRaLo Result


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-11/d96ef475-e7f5-4080-ac1a-f1d5c66d9f71/b12519930b290ed97e263ab972a2abc8cc0f3edbfe3aa29fafdfa9149943db4b.jpg)



(d) ReLoBRaLo Scaling


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-11/d96ef475-e7f5-4080-ac1a-f1d5c66d9f71/ae5c06d744ee703defa8e1b89eca030891f184b499df68f9855988fd8faf1b12.jpg)



Figure 9: Median of the log $L _ { 2 }$ loss over multiple training runs on the Helmholtz’s equation using (a) GradNorm, (b) Learning Rate Annealing and (c) ReLoBRaLo, as well as the mean and variance of scalings calculated by ReLoBRaLo (d).


and instead focuses on the more dominant governing equation. This is also reflected in the discrepancy between the training and validation loss: GradNorm and Learning Rate Annealing have a higher training loss than ReLoBRaLo, but still exceed at approximating the underlying function (cf. tab. VI). This triggered further investigation on the saudade and temperature parameters as described in the remainder of the next section. 

For the inverse Helmholtz problem setting, we select the wave number k to be learned for given data, which we obtained by sampling from the analytically known solution. Furthermore, we initialised $\mu = 0 . 5$ and tasked the network with approximating $k \ = \ 1$ . In the Helmholtz inverse problem setting similarly to the inverse Burgers problem, also just one optimiser was chosen for updating the network’s parameters θ together with the PDE parameter $\mu .$ . ReLoBRaLo also sets a new benchmark for Helmholtz’s inverse problem, both in accuracy as well as convergence speed, cf. fig. 10 and tab. VII. 


Convergence on inverse Helmholtz problem


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-11/d96ef475-e7f5-4080-ac1a-f1d5c66d9f71/34fd9169d68e759ebb13a1d957501319f8b19a59080966717e945e0990537bf2.jpg)



Figure 10: Approximation of the true PDE parameter value k (dashed line) in the inverse problem setting of Helmholtz’s equation. Reported values are the mean (solid line) and standard deviation (shaded area) of four independent runs.


## IX. Ablation and Sensitivity Study

The proposed ReLoBRaLo loss balancing scheme together with the PINNs architecture introduce many hyperparameters that have major influence on performance, efficiency and accuracy. In order to investigate the relations between hyperparameters and find their optimal combinations, we conduct an ablation and sensitivity study w.r.t. the temperature $T ,$ , exponential decay rate $\alpha ,$ and expected saudade $\mathbb { E } [ \rho ]$ and report its results in this section. 

Fig. 11 visualises the models’ sensitivity to the exponential decay rate α and the temperature T . A larger α causes the network to "remember" longer, while T controls how much the scalings "sheer out". Fig. 11(c) shows that Helmholtz’s equation benefits most from small values for $T ,$ which turn the balancing more aggressive. This is in line with the findings in the previous section (cf. sec. VIII-C), where we noted that the large difference in magnitudes between the terms in the loss function caused issues to ReLoBRaLo and that resolute balancing was necessary to avoid the boundary conditions to be neglected. In fact, we found the optimal $T$ to be $1 0 ^ { - 5 }$ (cf. tab. VIII). On the other hand, Burgers and Kirchhoff require smoother scalings with a tendency towards higher $T$ and α. It is worth noting that all three tasks benefit from the relaxation through the exponential decay, as setting $\alpha = 1$ always causes a deterioration of the model’s performance. 

<table><tr><td>Helmholtz</td><td></td><td>Baseline</td><td>GradNorm</td><td>LR anneal.</td><td>SoftAdapt</td><td>ReLoBRaLo</td></tr><tr><td rowspan="3">Forward</td><td>train f</td><td><eq>1.4 \cdot 10^{-2}</eq></td><td><eq>7.1 \cdot 10^{-2}</eq></td><td><eq>2.7 \cdot 10^{-1}</eq></td><td><eq>9.5 \cdot 10^{-3}</eq></td><td><eq>4.7 \cdot 10^{-3}</eq></td></tr><tr><td>val u</td><td><eq>7.1 \cdot 10^{-2}</eq></td><td><eq>5.6 \cdot 10^{-6}</eq></td><td><eq>1.4 \cdot 10^{-5}</eq></td><td><eq>1.6 \cdot 10^{-3}</eq></td><td><eq>2.6 \cdot 10^{-5}</eq></td></tr><tr><td>val std u</td><td><eq>8.1 \cdot 10^{-3}</eq></td><td><eq>1.9 \cdot 10^{-5}</eq></td><td><eq>7.6 \cdot 10^{-5}</eq></td><td><eq>1.5 \cdot 10^{-3}</eq></td><td><eq>8.2 \cdot 10^{-5}</eq></td></tr><tr><td>Inverse</td><td>val μ</td><td><eq>2.7 \cdot 10^{-3}</eq></td><td><eq>1.5 \cdot 10^{-1}</eq></td><td><eq>5.1 \cdot 10^{-2}</eq></td><td><eq>9.1 \cdot 10^{-2}</eq></td><td><eq>3.7 \cdot 10^{-4}</eq></td></tr><tr><td>k=1</td><td>std μ</td><td><eq>5.0 \cdot 10^{-2}</eq></td><td><eq>3.6 \cdot 10^{-1}</eq></td><td><eq>7.2 \cdot 10^{-2}</eq></td><td><eq>2.1 \cdot 10^{-2}</eq></td><td><eq>2.5 \cdot 10^{-4}</eq></td></tr></table>


Table VI: Comparison of the median $L _ { 2 }$ training and validation loss on Helmholtz’s equation against a baseline of manually chosen scalings. The reported values are the median over four independent runs with identical settings. Additionally, we report the standard deviation over the runs of the best performing model on the validation loss.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-11/d96ef475-e7f5-4080-ac1a-f1d5c66d9f71/1086ee4d8c650e63160a0df903cb9dc241b038e95bc7c374a93644bdb6dc889c.jpg)



Figure 11: Ablation of the model’s performance when varying $\tau$ and α with $\mathbb { E } [ \rho ] = 1$ . The reported values are the median of the log $L _ { 2 }$ loss over multiple training runs.


<table><tr><td></td><td>GradNorm</td><td>LR ann.</td><td>SoftAdapt</td><td>ReLoBRaLo</td></tr><tr><td><eq>\Delta T_{co}</eq> [s]</td><td>10.4</td><td>6.7</td><td>5.0</td><td>5.2</td></tr></table>


Table VII: Median computational overhead $\Delta T _ { c o }$ (in s) per 1’000 optimisation steps compared to using no balancing scheme (4.8s) on Helmholtz’s equation.


<table><tr><td>Hyperparameter</td><td>Burgers</td><td>Kirchhoff</td><td>Helmholtz</td></tr><tr><td>Learning Rate <eq>l_r</eq></td><td><eq>10^{-3}</eq></td><td><eq>10^{-3}</eq></td><td><eq>10^{-3}</eq></td></tr><tr><td>Layers <eq>d_K</eq></td><td>4</td><td>4</td><td>2</td></tr><tr><td>Neurons per Layer <eq>w_K</eq></td><td>256</td><td>360</td><td>256</td></tr><tr><td>Exponential Decay Rate <eq>\alpha</eq></td><td>0.999</td><td>0.999</td><td>0.99</td></tr><tr><td>Temperature <eq>\mathcal{T}</eq></td><td><eq>10^{-1}</eq></td><td><eq>10^{-2}</eq></td><td><eq>10^{-5}</eq></td></tr><tr><td>Expected Saudade <eq>\mathbb{E}[\rho]</eq></td><td>0.9999</td><td>0.9999</td><td>0.99</td></tr><tr><td>Activation function <eq>\sigma</eq></td><td><eq>tanh</eq></td><td><eq>tanh</eq></td><td><eq>tanh</eq></td></tr></table>


Table VIII: Final choices of hyperparameters for architecture and training settings.


However, the relaxation through the exponential decay induces a new trade-off between making the model remember longer and letting it adapt quickly to changes during training. We therefore study the effects of a random lookback through a Bernoulli random variable $\rho$ (saudade). It allows setting a lower value for $\alpha ,$ thus making the model more flexible, while occasionally "reminding" it of $\overline { { \mathcal { L } _ { i } ^ { ( 0 ) } } }$ 

Tab. IX summarises the change in performance when varying the expected saudade on all three experiments as a comparison. It is apparent that Helmholtz benefits more from frequent lookbacks, as it hits its best performance at $\mathbb { E } [ \rho ] = 0 . 9 9$ , whereas Burgers and Kirchhoff only require an expected lookback every 10’000 optimisation steps. Figs. 12 and 3 illustrate the effect of random lookbacks. While the stochasticity in the scaling factor increases and therefore makes them less interpretable, it increases the weight on the boundary conditions. The scaled contribution of the boundary conditions consequently leads to a better approximation of the underlying function uˆ. It is worth noting that the addition of the random lookback improves the accuracy on Helmholtz’ equation by more than an order of magnitude while having a lesser, albeit still significant effect on Burgers’ and Kirchhoff’s equations. 

<table><tr><td><eq>\mathbb{E}[\rho]</eq></td><td>Helmholtz</td><td>Burgers</td><td>Kirchhoff</td></tr><tr><td>0.0</td><td><eq>2.0\cdot10^{-3}</eq></td><td><eq>1.0\cdot10^{-3}</eq></td><td><eq>1.5\cdot10^{-09}</eq></td></tr><tr><td>0.5</td><td><eq>5.2\cdot10^{-5}</eq></td><td><eq>9.5\cdot10^{-3}</eq></td><td><eq>2.7\cdot10^{-09}</eq></td></tr><tr><td>0.9</td><td><eq>4.0\cdot10^{-5}</eq></td><td><eq>1.3\cdot10^{-3}</eq></td><td><eq>2.1\cdot10^{-09}</eq></td></tr><tr><td>0.99</td><td><eq>2.6\cdot10^{-5}</eq></td><td><eq>4.9\cdot10^{-4}</eq></td><td><eq>6.9\cdot10^{-10}</eq></td></tr><tr><td>0.999</td><td><eq>4.1\cdot10^{-5}</eq></td><td><eq>3.8\cdot10^{-4}</eq></td><td><eq>5.6\cdot10^{-10}</eq></td></tr><tr><td>0.9999</td><td><eq>1.2\cdot10^{-4}</eq></td><td><eq>1.4\cdot10^{-4}</eq></td><td><eq>4.0\cdot10^{-10}</eq></td></tr><tr><td>1</td><td><eq>8.1\cdot10^{-4}</eq></td><td><eq>4.7\cdot10^{-4}</eq></td><td><eq>7.4\cdot10^{-10}</eq></td></tr></table>


Table IX: Validation loss when varying the expected value of $\rho .$ The reported values are the median over three independent runs.


## X. Synopsis and Outlook

From previous work we observe, that a competitive relationship between physics loss items in the training of PINNs exists and potentially spoils training success, performance or efficiency. This paper investigated different methods aiming at adaptively balancing a loss function consisting of various, potentially conflicting objectives as it may arise in scalarised MOO in PINNs. We proposed a novel adaptive loss balancing method by (i) combining the best attributes of existing approaches, and (ii) introducing a saudade parameter ρ to occasionally incorporate historic loss contribution. This forms a new scheme called Relative Loss Balancing with Random Lookback (ReLoBRaLo) for selecting bespoke weights in order to combine multiple loss terms for the training of PINNs. The effectiveness and merits of using ReLoBRaLo was then demonstrated empirically by investigating several standard PDEs, including solving Helmholtz equation, Burgers’ equation and Kirchhoff plate bending equation, and considering both forward problems as well as inverse problems. Our computations showed that ReLoBRaLo is able to consistently outperform the baseline of existing scaling methods (GradNorm, Learning Rate Annealing, SoftAdapt) in terms of accuracy, while also being up to six times more computationally efficient (training epochs or wall-clock time). Finally, we showed that the adaptively chosen scalings λ can be inspected to learn about the PINNs training process and identify weak points. This allows to take informed decisions in order to improve the framework. 


(i) L2 Convergence


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-11/d96ef475-e7f5-4080-ac1a-f1d5c66d9f71/2eb7b4e76939fcf75c157e0d22514d7bc32f96c3c6f835a7b08a0ee3ae74f4f2.jpg)




[18] Guo, H., Zhuang, X., and Rabczuk, T. A Deep Collocation Method for the Bending Analysis of Kirchhoff Plate. arXiv e-prints (Feb. 2021), arXiv:2102.02617. 





shallow-networks avoid the curse of dimensionality: a review. International Journal of Automation and Computing 14, 5 (2017), 503–519. 




(ii) ReLoBRaLo Scaling




[19] Haghighat, E., Raissi, M., Moure, A., Gomez, H., and Juanes, R. A physics-informed deep learning framework for inversion and surrogate modeling in solid mechanics. Computer Methods in Applied Mechanics and Engineering 379 (2021), 113741. 





[47] Raissi, M. Deep hidden physics models: Deep learning of nonlinear partial differential equations. Journal of Machine Learning Research 19 (2018), 1–24. 



![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-11/d96ef475-e7f5-4080-ac1a-f1d5c66d9f71/f886ab48fc07b4f559deb25dcfa4f6335ef95964ab2e172f504ca47e23438c94.jpg)




[20] Heydari, A. A., Thompson, C. A., and Mehmood, A. SoftAdapt: Techniques for Adaptive Loss Weighting of Neural Networks with Multi-Part Loss Functions. arXiv e-prints (Dec. 2019), arXiv:1912.12355. 





[48] Raissi, M., Perdikaris, P., and Karniadakis, G. E. Physics Informed Deep Learning (Part II): Data-driven Discovery of Nonlinear Partial Differential Equations. arxiv.org (2017). 




Figure 12: Example of a single training process of ReLo-BRaLo on Helmholtz’s equation with $\alpha = 0 . 9 9 9$ , $\mathcal { T } = 1 0 ^ { - 4 }$ , $\mathbb { E } [ \rho ] = 0 . 9 9 9 9$ .




[21] Hornik, K., Stinchcombe, M., and White, H. Universal approximation of an unknown mapping and its derivatives using multilayer feedforward networks. Neural Networks 3, 5 (1990), 551–560. 





[49] Raissi, M., Perdikaris, P., and Karniadakis, G. E. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics 378 (2019), 686–707. 





[22] Hughes, T. The Finite Element Method: Linear Static and Dynamic Finite Element Analysis. Dover Civil and Mechanical Engineering. Dover Publications, 2012. 





[50] Raissi, M., Yazdani, A., and Karniadakis, G. E. Hidden fluid mechanics: Learning velocity and pressure fields from flow visualizations. Science 367, 6481 (2020), 1026–1030. 



Future research is concerned with inspection of performance, efficiency, robustness, and scalability of ReLoBRaLo to further PDE classes such as Navier-Stokes equations etc. The adoption of Sobolev Training with Sobolev norms or the incorporation of the Mixture-of-Experts approach [2] together with ReLoBRaLo may solve the drawback associated with the high costs involved in estimating the neural network solutions of PDEs. 



[23] Jagtap, A., and Karniadakis, G. Extended physics-informed neural networks (xpinns): A generalized space-time domain decomposition based deep learning framework for nonlinear partial differential equations. Communications in Computational Physics 28 (11 2020), 2002–2041. 





[51] Rajeswaran, A., Finn, C., Kakade, S., and Levine, S. Meta-Learning with Implicit Gradients. arXiv e-prints (Sept. 2019), arXiv:1909.04630. 



## References



[24] Jagtap, A., Kharazmi, E., and Karniadakis, G. Conservative physics-informed neural networks on discrete domains for conservation laws: Applications to forward and inverse problems. Computer Methods in Applied Mechanics and Engineering 365 (06 2020), 113028. 





[52] Ruchte, M., and Grabocka, J. Efficient multiobjective optimization for deep learning. ArXiv abs/2103.13392 (2021). 





[25] Jagtap, A. D., Kawaguchi, K., and Em Karniadakis, G. Locally adaptive activation functions with slope recovery for deep and physics-informed neural networks. Proceedings of the Royal Society of London Series A 476, 2239 (July 2020), 20200334. 





[53] Sener, O., and Koltun, V. Multi-task learning as multi-objective optimization. In Advances in Neural Information Processing Systems (2018), S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett, Eds., vol. 31, Curran Associates, Inc. 





[1] Bathe, K. Finite Element Procedures. No. pt. 2 in Finite Element Procedures. Prentice Hall, 1996. 





[26] Jagtap, A. D., Kawaguchi, K., and Karniadakis, G. E. Adaptive activation functions accelerate convergence in deep and physics-informed neural networks. Journal of Computational Physics 404 (Mar. 2020), 109136. 





[54] Shin, Y. On the Convergence of Physics Informed Neural Networks for Linear Second-Order Elliptic and Parabolic Type PDEs. Communications in Computational Physics 28, 5 (June 2020), 2042–2074. 





[2] Bischof, R., and Kraus, M. A. Mixture-of-expertsensemble meta-learning for physics-informed neural networks. In Proceedings of 33. Forum Bauinformatik (2022). 





[27] Jones, D. F., Mirrazavi, S. K., and Tamiz, M. Multi-objective meta-heuristics: An overview of the current state-of-the-art. European journal of operational research 137, 1 (2002), 1–9. 





[55] Shukla, K., Jagtap, A. D., and Karniadakis, G. E. Parallel Physics-Informed Neural Networks via Domain Decomposition. arXiv e-prints (Apr. 2021), arXiv:2104.10013. 





[3] Bonkile, M. P., Awasthi, A., Lakshmi, C., Mukundan, V., and Aswin, V. S. A systematic literature review of burgers’ equation with recent advances. Pramana 90, 6 (Apr 2018), 69. 





[28] Kendall, A., Gal, Y., and Cipolla, R. Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics. arXiv e-prints (May 2017), arXiv:1705.07115. 





[56] Sirignano, J., and Spiliopoulos, K. DGM: A deep learning algorithm for solving partial differential equations. Journal of Computational Physics 375 (2018), 1339–1364. 





[29] Kharazmi, E., Zhang, Z., and Karniadakis, G. E. hp-vpinns: Variational physics-informed neural networks with domain decomposition. Computer Methods in Applied Mechanics and Engineering 374 (2021), 113547. 





[57] Smith, G. Numerical Solutions of Partial Differential Equations: Finite Difference Methods, 3rd ed. Oxford University Press, New York. 





[4] Cai, S., Wang, Z., Chryssostomidis, C., and Karniadakis, G. E. Heat transfer prediction with unknown thermal boundary conditions using physicsinformed neural networks. In Fluids Engineering Division Summer Meeting (2020), vol. 83730, American Society of Mechanical Engineers, p. V003T05A054. 





[30] Kim, Y., Choi, Y., Widemann, D., and Zohdi, T. A fast and accurate physics-informed neural network 





[58] Smith-Miles, K. A. Cross-disciplinary perspectives on meta-learning for algorithm selection. ACM Comput. Surv. 41, 1 (Jan. 2009). 





[5] Caruana, R. Multitask learning. Machine Learning 28 (07 1997). 





[59] Snoek, J., Larochelle, H., and Adams, R. P. Practical bayesian optimization of machine learning algorithms. In Proceedings of the 25th International Conference on Neural Information Processing Systems - Volume 2 (Red Hook, NY, USA, 2012), NIPS’12, Curran Associates Inc., p. 2951–2959. 





[6] Chakraborty, S. Transfer learning based multifidelity physics informed deep neural network. Journal of Computational Physics 426 (2021), 109942. 





reduced order model with shallow masked autoencoder. arXiv e-prints (Sept. 2020), arXiv:2009.11990. 





[60] Sommerfeld, A. Partial differential equations in physics. Academic press, 1949. 





[7] Chang, K.-H. Chapter 17 - design optimization. In e-Design, K.-H. Chang, Ed. Academic Press, Boston, 2015, pp. 907–1000. 





[31] Kingma, D. P., and Ba, J. Adam: A Method for Stochastic Optimization. arXiv e-prints (Dec. 2014), arXiv:1412.6980. 





[8] Chen, Z., Badrinarayanan, V., Lee, C.-Y., and Rabinovich, A. GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks. arXiv e-prints (Nov. 2017), arXiv:1711.02257. 





[32] Kraus, M. A. Machine Learning Techniques for the Material Parameter Identification of Laminated Glass in the Intact and Post-Fracture State. PhD thesis, Universität der Bundeswehr München, 2019. 





[61] Son, H., Jang, J. W., Han, W. J., and Hwang, H. J. Sobolev training for the neural network solutions of pdes. arXiv preprint arXiv:2101.08932 (2021). 





[9] Czarnecki, W. M., Osindero, S., Jaderberg, M., Swirszcz, G., and Pascanu, R. Sobolev training for neural networks. In Advances in Neural Information Processing Systems (2017), vol. 2017-Decem, pp. 4279– 4288. 





[33] Kraus, M. A., and Drass, M. Artificial intelligence for structural glass engineering applications—overview, case studies and future potentials. Glass Structures & Engineering 5, 3 (2020), 247–285. 





[62] Theodoridis, S. Chapter 5 - online learning: the stochastic gradient descent family of algorithms. In Machine Learning (Second Edition), S. Theodoridis, Ed., second edition ed. Academic Press, 2020, pp. 179– 251. 





[10] EN. EN 1992-1-1 Eurocode 2: Design of concrete structures - Part 1-1: General ruels and rules for buildings (Brussels, 2005), CEN. 





[34] Lagaris, I., Likas, A., and Papageorgiou, D. Neural-network methods for boundary value problems with irregular boundaries. IEEE Transactions on Neural Networks 11, 5 (2000), 1041–1049. 





[63] Vlassis, N. N., Ma, R., and Sun, W. Geometric deep learning for computational mechanics part i: Anisotropic hyperelasticity. Computer Methods in Applied Mechanics and Engineering 371 (2020), 113299. 





[11] Finn, C., Abbeel, P., and Levine, S. Modelagnostic meta-learning for fast adaptation of deep networks. In Proceedings of the 34th International Conference on Machine Learning - Volume 70 (2017), ICML’17, JMLR.org, p. 1126–1135. 





[35] Lagaris, I. E., Likas, A., and Fotiadis, D. I. Artificial neural networks for solving ordinary and partial differential equations. IEEE transactions on neural networks 9, 5 (1998), 987–1000. 





[64] Vlassis, N. N., and Sun, W. Sobolev training of thermodynamic-informed neural networks for interpretable elasto-plasticity models with level set hardening. Computer Methods in Applied Mechanics and Engineering 377 (2021), 113695. 





[12] Finn, C., and Levine, S. Meta-Learning and Universality: Deep Representations and Gradient Descent can Approximate any Learning Algorithm. arXiv eprints (Oct. 2017), arXiv:1710.11622. 





[36] Liashchynskyi, P., and Liashchynskyi, P. Grid Search, Random Search, Genetic Algorithm: A Big Comparison for NAS. arXiv e-prints (Dec. 2019), arXiv:1912.06059. 





[65] Wang, S., Teng, Y., and Perdikaris, P. Understanding and mitigating gradient pathologies in physics-informed neural networks. arXiv e-prints (Jan. 2020), arXiv:2001.04536. 





[13] Finn, C., Xu, K., and Levine, S. Probabilistic Model-Agnostic Meta-Learning. arXiv e-prints (June 2018), arXiv:1806.02817. 





[37] Liu, X., Zhang, X., Peng, W., Zhou, W., and Yao, W. A novel meta-learning initialization method for physics-informed neural networks. arXiv preprint arXiv:2107.10991 (2021). 





[66] Wang, S., Yu, X., and Perdikaris, P. When and why PINNs fail to train: A neural tangent kernel perspective. arXiv e-prints (July 2020), arXiv:2007.14527. 





[14] Glorot, X., and Bengio, Y. Understanding the difficulty of training deep feedforward neural networks. In Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics (Chia Laguna Resort, Sardinia, Italy, 13–15 May 2010), Y. W. Teh and M. Titterington, Eds., vol. 9 of Proceedings of Machine Learning Research, PMLR, pp. 249–256. 





[38] Martins, J. R. R. A., and Ning, A. Engineering Design Optimization. Cambridge University Press, 2021. 





[67] Wight, C. L., and Zhao, J. Solving Allen-Cahn and Cahn-Hilliard Equations using the Adaptive Physics Informed Neural Networks. arXiv e-prints (July 2020), arXiv:2007.04542. 





[15] Goswami, S., Anitescu, C., Chakraborty, S., and Rabczuk, T. Transfer learning enhanced physics informed neural network for phase-field modeling of fracture. Theoretical and Applied Fracture Mechanics 106 (2020), 102447. 





[39] McClenny, L., and Braga-Neto, U. Self-Adaptive Physics-Informed Neural Networks using a Soft Attention Mechanism. arXiv e-prints (Sept. 2020), arXiv:2009.04544. 





[68] Xiang, Z., Peng, W., Zheng, X., Zhao, X., and Yao, W. Self-adaptive loss balanced physics-informed neural networks for the incompressible navier-stokes equations. arXiv preprint arXiv:2104.06217 (2021). 





[16] Grohs, P., Hornung, F., Jentzen, A., and Von Wurstemberger, P. A proof that artificial neural networks overcome the curse of dimensionality in the numerical approximation of blackscholes partial differential equations. arXiv preprint arXiv:1809.02362 (2018). 





[40] Meng, X., Li, Z., Zhang, D., and Karniadakis, G. E. Ppinn: Parareal physics-informed neural network for time-dependent pdes. Computer Methods in Applied Mechanics and Engineering 370 (2020), 113250. 





[69] Yuan, F.-G., Zargar, S. A., Chen, Q., and Wang, S. Machine learning for structural health monitoring: challenges and opportunities. In Sensors and Smart Structures Technologies for Civil, Mechanical, and Aerospace Systems 2020 (2020), vol. 11379, International Society for Optics and Photonics, p. 1137903. 





[17] Grossmann, C., Roos, H.-G., and Stynes, M. Numerical treatment of partial differential equations. 





[41] Mockus, J., Tiesis, V., and Zilinskas, A. The application of Bayesian methods for seeking the extremum, vol. 2. 09 2014, pp. 117–129. 





[70] Zhang, C., Liao, Q., Rakhlin, A., Miranda, B., Golowich, N., and Poggio, T. Theory of Deep Learning IIb: Optimization Properties of SGD. arXiv e-prints (Jan. 2018), arXiv:1801.02254. 





[42] Nichol, A., Achiam, J., and Schulman, J. On First-Order Meta-Learning Algorithms. arXiv e-prints (Mar. 2018), arXiv:1803.02999. 





[43] Nogueira, F. Bayesian Optimization: Open source constrained global optimization tool for Python, 2014– 





[44] Orlandi, P. The Burgers equation. Springer Netherlands, Dordrecht, 2000, pp. 40–50. 





[45] Peng, W., Zhou, W., Zhang, J., and Yao, W. Accelerating Physics-Informed Neural Network Training with Prior Dictionaries. arXiv e-prints (Apr. 2020), arXiv:2004.08151. 





[46] Poggio, T., Mhaskar, H., Rosasco, L., Miranda, B., and Liao, Q. Why and when can deep-but not 

