# ViscoelasticNet: A physics informed neural network framework for stress discovery and model selection

Sukirt Thakur1, Maziar Raissi2, and Arezoo M. Ardekani1 

# Abstract

Viscoelastic fluids are a class of fluids that exhibit both viscous and elastic nature. Modelling such fluids requires constitutive equations for the stress, and choosing the most appropriate constitutive relationship can be difficult. We present viscoelasticNet, a physics-informed deep learning framework that uses the velocity flow field to select the constitutive model and learn the stress field. Our framework requires data only for the velocity field, initial & boundary conditions for the stress tensor, and the boundary condition for the pressure field. Using this information, we learn the model parameters, the pressure field, and the stress tensor. This work considers three commonly used non-linear viscoelastic models: Oldroyd-B, Giesekus, and linear Phan-Thien-Tanner (PTT). We demonstrate that our framework works well with noisy and sparse data. Our framework can be combined with velocity fields acquired from experimental techniques like particle image velocimetry to get the pressure & stress fields and model parameters for the constitutive equation. Once the model has been discovered using viscoelasticNet, the fluid can be simulated and modeled for further applications. 

Keywords— Physics informed neural networks, Viscoelastic flow, Deep learning, Inverse modelling 

# 1 Introduction

Fluids can be categorized based on their response to the strain rate or the change in deformation with respect to time. For fluids that obey Newton’s law of viscosity, the viscous stress at every point correlates linearly with the local strain rate. Numerous fluids, called non-Newtonian fluids, exhibit complex rheological behavior which deviates from Newton’s law of viscosity. We can classify non-Newtonian fluids as inelastic, linear-viscoelastic, and non-linear viscoelastic fluids. Viscoelastic fluids are a class of non-Newtonian fluids that exhibit viscous and elastic characteristics when subjected to deformation. These fluids are pertinent to various biological and industrial processes such as fertilization [1, 2], the collective motion of microorganisms [3, 4], and oil recovery [5, 6]. 

The conservation of mass and momentum governs all fluid equations. The forces acting on the fluid are obtained for Newtonian fluids, assuming a linear correlation between stress and strain. However, viscoelastic fluids have both elastic and viscous characteristics. Hence, we need to solve a constitutive equation for stress along with the continuity and momentum equations. While linear viscoelastic models work well for small deformations, constitutive models that capture the non-linearity between stress and strain are required for large deformations. These non-linear viscoelastic models can describe complex phenomena like shear thinning and extensional thickening. Numerical methods like finite difference, finite elements, and finite volume are often required to obtain the stress field using these constitutive equations. However, non-linear viscoelastic models are often computationally demanding and require numerical tricks to ensure stability [7, 8, 9]. Moreover, selecting the most appropriate model for the fluid of interest can be challenging. 

Deep learning-based frameworks have helped solve challenging problems in various fields. These include biomedical imaging [10, 11], computer vision [12, 13], and natural language processing [14, 15]. There is growing interest in leveraging these techniques to understand and model biological and engineering systems. Machine learning algorithms have been used for problems in fluid mechanics for surrogate modeling, design optimization, and reduced order and closure models [16, 17, 18, 19]. A deep neural network has been used to model viscoelastic properties from observed displacement data – as a PDE-constrained optimization challenge [?]. However, many of these algorithms are data-intensive, and acquiring data at scale for engineering systems is often expensive. 

Physics-informed neural networks (PINNs) have emerged as a powerful tool in this context. PINNs [20, 21], supervised learning frameworks with embedded physics, allow us to train massive neural networks with relatively small training datasets. PINNs achieve this data efficiency by using the governing equations to regularize the optimization of the neural network’s parameters, enabling them to generalize even when few examples are available. While the most popular neural network architecture used for PINNs is a vanilla feed-forward neural network, researchers have explored other architectures in the literature. PINNs have been extended to use multiple feed-forward networks [22, 23], convolution neural networks [24, 25], recurrent neural networks [26, 27], and Bayesian neural networks [28]. Researchers have used PINNs to help solve various forward and inverse problems in fluid mechanics [29, 30, 31]. Hidden fluid mechanics (HFM) [32], a physics-informed deep learning framework, has been used to extract quantitative information from flow visualization. PINN-based frameworks have been used for solving Reynolds-averaged Navier Stokes equations [33], for modeling porous media flows [34], and to solve inverse problems of three-dimensional supersonic and biomedical flows [35]. Recently, a non-Newtonian PINNs-based framework was used for solving complex fluid systems [36]. 

Physics-informed neural networks (PINNs) can be extended along several dimensions. These include: 1) more complex physics (i.e., equations), 2) more complex geometries, 3) better loss functions, 4) better architectures, and 5) better training processes. We are making contributions along dimensions 1, 3, and 5. In this work, we present viscoelasticNet, a physics-informed neural networks-based framework that uses the velocity flow field to select the viscoelastic constitutive model and learn the stress field. We consider three commonly used non-linear viscoelastic models: the Oldroyd-B [37], Giesekus [38], and Linear PTT [39]. We combine the equations for these models into a single general equation. We generate numerical data for each model mentioned above and employ our framework to learn the model parameters. Through this process, we showcase the capability of our framework to evaluate and select the most suitable model from the three considered models based on the learned parameters. We also learn the pressure field and the stress tensor for the flow. The observables for our method are only the velocity field, the boundary and initial conditions for the stress field, and the boundary conditions for the pressure field. Hence, our method can be combined with experimentally acquired velocity fields to get the stress and pressure fields and select the viscoelastic constitutive equation for the fluid. We discuss the problem setup and methodology in section 2. We test our framework using the geometry of two-dimensional stenosis and a cross-slot geometry. We tested our framework for noise and sparsity in the velocity field using the stenosis geometry, and we used cross-slot geometry to carry out further tests on the effect of variation in parameters and the boundary conditions. Finally, we discuss the results in section ?? and provide some concluding remarks on our study in section ??. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-27/b57158d5-9d9e-4be4-9032-68fa98ae70ae/ee268d9b5dd0c6ddde30384aad2a981e8d6ab7ee1896091596c56bf58894aa61.jpg)



Figure 1: A schematic of the neural network set up to learn the stress field and parameters in eq. (13) by minimizing the loss function presented in eq. (20). We use two fully connected neural networks to estimate the general constitutive equation’s stress and parameters. The network for velocity has an ivory color, while the network for stress has a green color, as shown in the figure. We use automatic differentiation to calculate the losses that we describe in section 2. We denote the identity operator by I and use automatic differentiation to compute the differential operators ∂t, ∂x, ∂y.


# 2 Problem setup and methodology

# 2.1 Fluid motion equations

Consider an incompressible fluid under isothermal, single-phase, transient conditions in a domain $\Omega \subset \mathbb { R } ^ { d }$ with boundary $\partial \Omega = \Gamma _ { D } \cup \Gamma _ { N }$ . The parameters $\Gamma _ { D }$ and $\Gamma _ { N }$ are portions of the boundary, respectively, where a Dirichlet and a Neumann boundary condition is applied, and d is the dimension. The following equations give the mass conservation and momentum balance in the absence of any body force 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-27/b57158d5-9d9e-4be4-9032-68fa98ae70ae/39634243e901ae42b9d3a0f46eb29f032da6571fa81bc0ef6ef82710cc56d625.jpg)



Figure 2: A schematic of the neural network set up to learn the pressure field and the viscosity by minimizing the loss function presented in eq. (24). The figure shows three neural networks with different colors: ivory for velocity, green for stress, and purple for pressure. We fix the parameters of the stress network when we solve for the pressure. We calculate the losses using automatic differentiation, as we explain in section 2.


$$
\nabla \cdot \boldsymbol {u} = 0, \tag {1}
$$

$$
\rho \left(\frac {\partial \boldsymbol {u}}{\partial t} + \boldsymbol {u} \cdot \nabla \boldsymbol {u}\right) = - \nabla p + \nabla \cdot \boldsymbol {\tau} ^ {\prime}, \tag {2}
$$

where $\rho$ is the density of the fluid u is the velocity vector, t is the time, p is the pressure, and $\tau ^ { \prime }$ is the stress tensor. As for the boundaries, we have 

$$
\left\{ \begin{array}{l l} \mathbf {u} = \mathbf {g} & \text { on } \Gamma_ {D} \times (0, T) \\ \boldsymbol {\tau} ^ {\prime} (\mathbf {u}, p) \hat {\mathbf {n}} = \mathbf {h} & \text { on } \Gamma_ {N} \times (0, T) \\ \mathbf {u} (0) = \mathbf {u} _ {0} & \text { in } \Omega \times \{0 \}, \end{array} \right. \tag {3}
$$

where nˆ is the outward directed unit normal, the functions g and h are given the Dirichlet and Neumann boundary data, respectively, and $\mathbf { u } _ { 0 }$ is the initial condition. This work will represent scalars by non-bold characters, vectors by bold lowercase characters, and matrices by bold uppercase characters. 

# 2.2 Rheological constitutive model

The stress tensor $\tau ^ { \prime }$ in $\mathrm { e q . }$ (2) for viscoelastic fluids is often split into solvent and polymeric parts, 

$$
\boldsymbol {\tau} ^ {\prime} = \boldsymbol {\tau} ^ {s} + \boldsymbol {\tau}. \tag {4}
$$

We need a constitutive relation for the solvent and polymeric stress to have a wellposed problem. For a significant number of models, we can write the constitutive equations in the following form 

$$
\boldsymbol {\tau} ^ {s} = \eta_ {s} (\nabla \boldsymbol {u} + \nabla \boldsymbol {u} ^ {T}), \tag {5}
$$

$$
f (\boldsymbol {\tau}) \boldsymbol {\tau} + \lambda \overset {\nabla} {\boldsymbol {\tau}} + \boldsymbol {h} (\boldsymbol {\tau}) = \eta_ {p} (\nabla \boldsymbol {u} + \nabla \boldsymbol {u} ^ {T}), \tag {6}
$$

where we denote the solvent viscosity by $\eta _ { s }$ , the polymeric viscosity by $\eta _ { p } ,$ , the relaxation time by $\lambda ,$ the shear rate by ${ \dot { \gamma } } , f ( \tau )$ is a scalar-valued function, $h ( \tau )$ is a tensor-valued function and $\begin{array} { l } { \displaystyle { \frac { \nabla } { \tau } } } \end{array}$ is the upper convected time derivative which is defined as 

$$
\boldsymbol {\nabla} \boldsymbol {\tau} = \frac {D \boldsymbol {\tau}}{D t} - (\nabla \boldsymbol {u}) ^ {T} \cdot \boldsymbol {\tau} - \boldsymbol {\tau} \cdot (\nabla \boldsymbol {u}), \tag {7}
$$

where 

$$
\frac {D \boldsymbol {\tau}}{D t} = \frac {\partial \boldsymbol {\tau}}{\partial t} + \boldsymbol {u} \cdot \nabla \boldsymbol {\tau} \tag {8}
$$

is the material derivative. The conservation of angular momentum principle implies that the polymetric stress tensor τ is symmetric. Hence, we define the stress tensor τ in two dimensions by three independent parameters 

$$
\boldsymbol {\tau} = \left[ \begin{array}{c c} \tau^ {x x} & \tau^ {x y} \\ \tau^ {x y} & \tau^ {y y} \end{array} \right], \tag {9}
$$

where $\tau ^ { x x }$ and $\tau ^ { y y }$ are the orthogonal normal stresses and $\tau ^ { x y }$ is the orthogonal shear stress. This work considers the Oldroyd-B [37], Giesekus [38], and Linear PTT [39] models. The respective constitutive equations for the Oldroyd-B, Giesekus, and Linear PTT models are given by 

$$
\boldsymbol {\tau} + \lambda \overset {\nabla} {\boldsymbol {\tau}} = \eta_ {p} (\nabla \boldsymbol {u} + \nabla \boldsymbol {u} ^ {T}), \tag {10}
$$

$$
\boldsymbol {\tau} + \lambda \overset {\nabla} {\boldsymbol {\tau}} + \alpha \frac {\lambda}{\eta_ {p}} (\boldsymbol {\tau} \cdot \boldsymbol {\tau}) = \eta_ {p} (\nabla \boldsymbol {u} + \nabla \boldsymbol {u} ^ {T}), \tag {11}
$$

and 

$$
\left(1 + \frac {\epsilon \lambda}{\eta_ {p}} t r (\boldsymbol {\tau})\right) \boldsymbol {\tau} + \lambda \overline {{\boldsymbol {\tau}}} = \eta_ {p} (\nabla \boldsymbol {u} + \nabla \boldsymbol {u} ^ {T}), \tag {12}
$$

where $t r ( \tau )$ denotes the trace of the stress tensor, ϵ represents the extensibility parameter and α is the mobility parameter. We write the following general form equation 

$$
\left(1 + \frac {\epsilon \lambda}{\eta_ {p}} t r (\boldsymbol {\tau})\right) \boldsymbol {\tau} + \lambda \overline {{\boldsymbol {\tau}}} + \alpha \frac {\lambda}{\eta_ {p}} (\boldsymbol {\tau} \cdot \boldsymbol {\tau}) = \eta_ {p} (\nabla \boldsymbol {u} + \nabla \boldsymbol {u} ^ {T}), \tag {13}
$$

which we use to represent the Oldroyd-B, Giesekus, and Linear PTT models. Equations 10, 11, 12, and 13 are special cases of Eq. 6. As shown in table 1, learning the values for the extensibility parameter (ϵ) and the mobility parameter (α) can help us select the constitutive equation that best describes the flow. If ϵ and α equal zero, the Oldroyd-B model can describe the flow. Similarly, if ϵ or α are non-zero, the flow can be described using the linear PTT and Giesekus model, respectively. If the learned values of both ϵ and α are non-zero, it implies that these three constitutive equations cannot describe the fluid. In this work, we demonstrate how the most appropriate model can be determined among these options based on the values of the learned parameters ϵ and α. 


Table 1: The list of parameters in eq. (13) to represent the Oldroyd-B, Giesekus, and Linear PTT models.


<table><tr><td>Model</td><td>ε</td><td>α</td></tr><tr><td>Oldroyd</td><td>0</td><td>0</td></tr><tr><td>Gieseukus</td><td>0</td><td>≠0</td></tr><tr><td>Linear PTT</td><td>≠0</td><td>0</td></tr></table>

# 2.3 Physics informed neural networks

We develop a physics-informed neural network-based framework called viscoelastic-Net, which combines the information available in the velocity field, the Navier-Stokes equation, and the general form of the constitutive equation, eq. (13). The objective is to learn the parameters of the constitutive equation while simultaneously solving the forward problem to obtain the stress field. We consider the velocity field ${ \pmb u } ( t , { \pmb x } ) = [ u ( t , { \pmb x } ) , v ( t , { \pmb x } ) ]$ of an incompressible isothermal flow of a viscoelastic fluid, where $\pmb { x } = ( x , y )$ . We observe N data points of time-space coordinates $( t ^ { n } , x ^ { n } , y ^ { n } )$ and the velocity of the fluid corresponding to these points $( u ^ { n } , v ^ { n } )$ where $n = 1 , \ldots , N$ . Given such scattered spatiotemporal data, we are interested in the discovery of the components of the stress tensor ${ \boldsymbol { \tau } } ( t , { \boldsymbol { \mathbf { x } } } )$ as well as their governing equation by determining the parameters $\epsilon , \lambda , \alpha , \eta _ { p }$ and ηs in eq. (13). Our setup has no input data on the pressure field and the stress tensor except for the initial and boundary conditions. In our setup, we treat the x and y components of the velocity, the value of the stress field at the first time step (initial value), the stress field at the inlet, and the value of the pressure field at the outlet as the observable. We approximate the functions $( t , \pmb { x } ) \longmapsto ( \sigma ^ { x x } , \sigma ^ { x y } , \sigma ^ { y y } )$ , (t, x) 7−→ ψ and (t, x) 7−→ p using three deep neural networks with parameters θ, ϕ and κ called the stress network, the velocity network and the pressure network, respectively. For the x-component of velocity u(t, x) and y-component v(t, x), we define 

$$
u = \psi_ {y}, v = - \psi_ {x}, \tag {14}
$$

for a scalar $\psi ( t , { \pmb x } )$ and the subscripts represent partial derivatives. Defining the velocity field using a vector potential ψ = (0, 0, ψ) allows us to make the velocity field divergence free by construction, as we define u = ∇ × ψ. This approach can be extended to three dimensions as well. The velocity field then automatically satisfies the continuity equation, eq. (1). We utilize a neural network to represent the velocity field because it enables us to compute derivatives of the velocity components with respect to the inputs, facilitating the calculation of residuals for the equations. While derivatives could be approximated using numerical methods like the finite difference approach, employing a neural network for the velocity field leverages automatic differentiation, which offers superior accuracy, efficiency, and stability compared to other numerical techniques. We decouple the momentum equations from the constitutive equations for the polymeric stress and sequentially solve them. We chose a separate network for pressure as, in our experience, this setup works better with our decoupled sequential approach, and it is a fairly common technique employed in computational fluid dynamics to decouple pressure from the momentum equations. We define the mean squared error loss for regression over the velocity field as 

$$
L _ {v e l} (\phi) = \mathbb {E} _ {(t, \boldsymbol {x}, \boldsymbol {u})} \left[ \frac {| \boldsymbol {u} (t , \boldsymbol {x} ; \phi) - \boldsymbol {u} | ^ {2}}{\sigma_ {\boldsymbol {u}} {} ^ {2}} \right], \tag {15}
$$

where u is the reference velocity field, ${ \pmb u } ( t , { \pmb x } ; \phi )$ is the prediction from the network, and $\sigma _ { u }$ is the standard deviation of the reference velocity field, and E denotes the expectation approximated by the population mean (i.e., mean of the observations $t _ { n } , x _ { n } , y _ { n } , u _ { n } , v _ { n }$ where $n = 1 , \ldots , M$ for M observations). Since we are also solving the forward problem of learning the stress field, initial and boundary conditions on the stress field are required. We enforce the initial condition using the loss function 

$$
L _ {i n i t} (\theta) = \mathbb {E} _ {(t ^ {i n i t}, \boldsymbol {x} ^ {i n i t}, \boldsymbol {\tau} ^ {i n i t})} \left[ \frac {| \boldsymbol {\tau} (t ^ {i n i t} , \boldsymbol {x} ^ {i n i t} ; \theta) - \boldsymbol {\tau} ^ {i n i t} | ^ {2}}{\sigma_ {\boldsymbol {\tau}} {} ^ {2}} \right], \tag {16}
$$

where $t ^ { i n i t } , \pmb { x } ^ { i n i t }$ is the spatio-temporal point cloud at the initial timestep, $\tau ^ { i n i t }$ is the stress field at the first time step $t ^ { i n i t }$ and $\sigma _ { \tau }$ is standard deviation of $\tau ^ { i n i t }$ . We define $t ^ { i n i t }$ as the initial timestep. It can be 0, or any other value that the user chooses which corresponds to the $\tau ^ { i n i t }$ being considered. For brevity, we define $\Pi = ( \lambda , \epsilon , \alpha , \eta _ { p } )$ . Now, let 

$$
\begin{array}{l} \mathbf {f} (t, \boldsymbol {x}; \theta , \phi , \Pi) = \left(\frac {1}{\lambda} + \frac {\epsilon}{\eta_ {p}} t r (\boldsymbol {\tau})\right) \boldsymbol {\tau} + \boldsymbol {u} \cdot \nabla \boldsymbol {\tau} - (\nabla \boldsymbol {u}) ^ {T} \cdot \boldsymbol {\tau} - \boldsymbol {\tau} \cdot (\nabla \boldsymbol {u}) \tag {17} \\ + \frac {\alpha}{\eta_ {p}} (\pmb {\tau} \cdot \pmb {\tau}) - \frac {\eta_ {p}}{\lambda} (\nabla \pmb {u} + \nabla \pmb {u} ^ {T}). \\ \end{array}
$$

Eq. (17) represents the value of $\frac { \partial \tau } { \partial t }$ in eq. (13), and this definition allows us to use the backward Euler discretization to construct a “physics-informed” network. The output of the feed-forward networks will be called “physics uninformed” in the rest of the text and denoted with a superscript $\mathrm { ^ { * } p u } ^ { \dag }$ . We then create a physics-informed neural network using the backward Euler discretization 

$$
\boldsymbol {\tau} ^ {p i} (t, \boldsymbol {x}; \Delta t, \theta , \phi , \Pi) = \boldsymbol {\tau} ^ {p u} (t + \Delta t, \boldsymbol {x}; \theta) + \Delta t \mathbf {f} (t + \Delta t, \boldsymbol {x}; \theta , \phi , \Pi), \tag {18}
$$

where the superscript $^ { \ast } \mathrm { p i } ^ { \prime \prime }$ is used to denote “physics-informed”. Since the physicsinformed and uninformed networks evaluate the stress at the same point $( t , \pmb { x } )$ , they must be consistent. We enforce this using a consistency loss 

$$
L _ {\text { consistency }} (\theta ; \Delta t, \phi , \Pi) = \mathbb {E} _ {(t, \boldsymbol {x})} \left[ \frac {| \boldsymbol {\tau} ^ {p i} (t , \boldsymbol {x} ; \Delta t , \theta , \phi , \Pi) - \boldsymbol {\tau} ^ {p u} (t , \boldsymbol {x} ; \theta) | ^ {2}}{\sigma_ {\boldsymbol {\tau}} {} ^ {2}} \right], \tag {19}
$$

In this work, we utilize backward Euler time-stepping to determine the relative weights for the loss terms based on the standard deviation of the available data. Using the standard deviation provides us with an equation-specific scale. We add the case-specific Neumann and Dirichlet boundary conditions for stress, $L _ { N e u m a n n } ( \theta )$ and $L _ { D i r i c h l e t } ( \theta )$ , respectively. The parameters θ and $\phi$ are then optimized along with $\lambda , \epsilon ,$ and $\eta _ { p }$ to 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-27/b57158d5-9d9e-4be4-9032-68fa98ae70ae/9ab3a6ad3b1e1fa29260d804859e3136579e6d9c926fb3c34afde029a44ebdf7.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-27/b57158d5-9d9e-4be4-9032-68fa98ae70ae/5288dc51df871b5a431242d7aa880a750ef85ab59b2fd72557648c831bcf7e67.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-27/b57158d5-9d9e-4be4-9032-68fa98ae70ae/2cdb67e8eec09f745b49ae49dc300c12dcfa320e55fc75ce2e68e1141f59cdb0.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-27/b57158d5-9d9e-4be4-9032-68fa98ae70ae/1e4dbb5eb9e67b90beacd31e26cb89233ba5c9092e231008f32f02bd3faebaf2.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-27/b57158d5-9d9e-4be4-9032-68fa98ae70ae/270ab591e3ea5f532c369aaa46461a6bfb1a939d8a99dee5412948675f7721a2.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-27/b57158d5-9d9e-4be4-9032-68fa98ae70ae/85c589a04b95681de194f503326819617458b3556d089e4e826ad4e45d0d1663.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-27/b57158d5-9d9e-4be4-9032-68fa98ae70ae/b5daa8f187d4578fb63ef69b75bc017518f9699652f0b3dcbae7d5805c651b69.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-27/b57158d5-9d9e-4be4-9032-68fa98ae70ae/d653ff28ad4cf96bbac5dcd0e09b57d61ea2b24a7bb73c8918fda3248d4e3725.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-27/b57158d5-9d9e-4be4-9032-68fa98ae70ae/d1436014d5d570ace639e180ef224497debb606f8ad0aaa95e186dad6748ab4a.jpg)



Figure 3: Relative errors between the predictions of the model and the corresponding reference components of the stress field across time steps for (A) Oldroyd (B) Linear PTT (C) Giesekus constitutive models.



## Algorithm 1 The algorithm for viscoelasticNet


Input: Spatio-temporal point clouds, IC and BCs on stress
1: $\theta,\phi\leftarrow\theta^{0},\phi^{0}$ ▷ Initialize the neural network parameters
2: for iteration = 1, 2, … do ▷ Loop till the number of iterations
3: Compute $L_{stress}(\theta,\phi)$ 4: Update learning rate
5: $\theta,\phi,\alpha,\epsilon,\lambda,\eta_{p}\leftarrow\text{Optimizer}(L_{stress}(\theta,\phi),\text{learning rate})$ 6: end for
7: Freeze the optimized parameter $\theta'$ 8: for iteration = 1, 2, … do ▷ Loop till the number of iterations
9: Compute $K_{pressure}(\phi,\kappa)$ 10: Update learning rate
11: $\phi,\kappa,\eta_{s}\leftarrow\text{Optimizer}(K_{pressure}(\phi,\kappa),\text{learning rate})$ 12: end for
Output: $\theta',\phi',\kappa',\alpha',\epsilon',\lambda',\eta_{p}',\eta_{s}'$ ▷ Optimized parameters 

minimize the following combined loss 

$$
\begin{array}{r l} L _ {\text { stress }} (\theta , \phi ; \Pi) & = L _ {\text { vel }} (\phi) + L _ {\text { initial }} (\theta) + L _ {\text { Neumann }} (\theta) + \\ & \quad I _ {\text { stress }} (\theta) + I _ {\text { initial }} (\theta ; \phi , \Pi) \end{array} \tag {20}
$$

$$
L _ {D i r i c h l e t} (\theta) + L _ {c o n s i s t e n c y} (\theta ; \phi , \Pi).
$$

We show the schematic of the network in figure 2, and the algorithm for viscoelasticNet is explained using the pseudo algorithm 1. Regularization is a practice used to avoid overfitting in machine learning. We use our prior knowledge of the governing equations to regularize the optimization process of the neural network parameters, as $L _ { ☉ }$ consistency penalizes solutions that do not satisfy the governing equation. In this work, we utilize backward Euler time-stepping to determine the relative weights for the loss terms based on the standard deviation of the available data. We experimented with other techniques to obtain the loss weights, including assigning gradient-based weights and applying Lagrange multipliers. However, we found that the backward Euler method performed best for our application. Since we are sequentially solving the problem, we freeze the optimized parameters θ′ of the neural network for the stress while solving for pressure. We split the momentum equation, eq. (2), into two parts, one which can be directly computed from the observables and the second which has unknown components. The convective part of the momentum equations is given by 

$$
\boldsymbol {g} ^ {L} (\boldsymbol {u}) = \frac {\partial \boldsymbol {u}}{\partial t} + \boldsymbol {u} \cdot \nabla \boldsymbol {u}, \tag {21}
$$

and 

$$
\boldsymbol {g} ^ {R} (p; \boldsymbol {u}, \boldsymbol {\tau}, \eta_ {s}) = - \nabla p + \eta_ {s} (\nabla^ {2} \boldsymbol {u}) + \nabla \cdot \boldsymbol {\tau}. \tag {22}
$$

We calculate the standard deviation of $\pmb { g } ^ { L }$ as $\sigma _ { g }$ . We then enforce the momentum equations, eq. (2), using 

$$
K _ {m} (\kappa ; \phi , \eta_ {s}) = \mathbb {E} _ {(t, \boldsymbol {x})} \left[ \frac {\left| \begin{array}{c c} \boldsymbol {g} ^ {L} (\boldsymbol {u} (t , \boldsymbol {x} ; \phi)) - \boldsymbol {g} ^ {R} (p (t , \boldsymbol {x} ; \kappa) ; \boldsymbol {u} (t , \boldsymbol {x} ; \phi) , & \\ & \boldsymbol {\tau} (t , \boldsymbol {x} ; \theta^ {\prime}) , \eta_ {s}) \end{array} \right| ^ {2}}{\sigma_ {g} ^ {2}} \right]. \tag {23}
$$

We add the case-specific Neumman and Dirichlet boundary conditions for pressure $( K _ { N e u m a n n } ( \kappa )$ and $K _ { D i r i c h l e t } ( \kappa )$ , respectively) and optimize the parameters ϕ and κ along with ηs using the following combined loss 

$$
K _ {p r e s s u r e} (\phi , \kappa ; \eta_ {s}) = L _ {v e l} (\phi) + K _ {m o m} (\kappa ; \phi , \eta_ {s}) + K _ {N e u m a n n} (\kappa) + K _ {D i r i c h l e t} (\kappa). (2 4)
$$

# 3 Results

# 3.1 Stenosis

We consider a two-dimensional stenosis geometry, as shown in Fig. 5. We used RheoTool [40], an OpenFOAM [41] based open source software developed by Favero et al. [42] to generate the training and reference data sets. RheoTool uses the finite volume method to discretize the equations. It uses the both-side-diffusion technique to increase the ellipticity, stabilizing the momentum equation. We use the log-confirmation tensor approach to tackle the numerical instabilities in the polymeric stress. More details on the solver and the validation for the code can be found here [40, 42]. We 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-27/b57158d5-9d9e-4be4-9032-68fa98ae70ae/925eb402316ceff9efe6cb602076b9925ac844f7839ea800511c7ac2b0fa9fbe.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-27/b57158d5-9d9e-4be4-9032-68fa98ae70ae/4d6a861a23f4803262b2b4172e7bbe2e9b863c2ecdf5c70c29f46ee423af4243.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-27/b57158d5-9d9e-4be4-9032-68fa98ae70ae/2ccd6a6c546e61cbaf9990a40285830630188a60221ddb3087019b5808126bae.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-27/b57158d5-9d9e-4be4-9032-68fa98ae70ae/59022bf4a24680c773b09e594b1b78216aca137492dc6d74e1863ee3baa0aa0f.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-27/b57158d5-9d9e-4be4-9032-68fa98ae70ae/bb6e45924f81544bad264922416dd7f6f9d714abe09822d1719a8860933a845f.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-27/b57158d5-9d9e-4be4-9032-68fa98ae70ae/634c50cae686d6e378856ce03488fd4b89f2749b97f7c12582347b0350a364fb.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-27/b57158d5-9d9e-4be4-9032-68fa98ae70ae/ffb226aa71c3b4b67cf10a0855f21db5057ef3ec3b1c0e103210b790e430a842.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-27/b57158d5-9d9e-4be4-9032-68fa98ae70ae/2035675105d0173e715a717998a5fae6215afefe2bc81e9578182a7eb8888723.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-27/b57158d5-9d9e-4be4-9032-68fa98ae70ae/1c05fa7fc67a43d0212efec3decd0649109b6494260e1bbdcc6da08b26cf5e4e.jpg)



Figure 4: Relative errors between the predictions of the model and the corresponding reference velocity and pressure fields across the time steps for (A) Oldroyd (B) Linear PTT (C) Giesekus constitutive models.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-27/b57158d5-9d9e-4be4-9032-68fa98ae70ae/a43ba65eaa6b4aa85060e13d806a7ed588adf25ca7bbd20b2dfb70e750450e1f.jpg)



Figure 5: A two-dimensional stenosis. We show the domain walls using solid blue curves, the inlet with a dotted orange line, and the outlet with a dashed green line. The lower wall of the stenosis is highlighted using a black line, and we plot the stress on the lower wall in Fig. 6. The narrowest part of the throat of the stenosis is highlighted with a red line. We plot the stress in this region in Fig. 7.



(A)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-27/b57158d5-9d9e-4be4-9032-68fa98ae70ae/ad0ce540252fc7a6beb1daf6773b6369b63f7c4b4ad017b94cee186bf0a64fdf.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-27/b57158d5-9d9e-4be4-9032-68fa98ae70ae/939aeac901f6c65004dd426c8ae8d5cda07790ff17491113586a15531433f132.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-27/b57158d5-9d9e-4be4-9032-68fa98ae70ae/81e04e42632a1145772f315b88787771a3656f29ceba5926ccf85072825d2cca.jpg)



(B)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-27/b57158d5-9d9e-4be4-9032-68fa98ae70ae/1926aeb86d026ce2154c6e3cee1558d7385f9a6d94f2cf689b0de24741db127c.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-27/b57158d5-9d9e-4be4-9032-68fa98ae70ae/d95719fe88030fe72db07857837bc1c903c16d2867d70799c83653a76d005e7d.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-27/b57158d5-9d9e-4be4-9032-68fa98ae70ae/c628b2497a5e1e3416f76f3199ae144e007d8e1d50319f5d176ebf463fb5a56f.jpg)



(C）


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-27/b57158d5-9d9e-4be4-9032-68fa98ae70ae/6624501c6d1daf78c61ef76de4dfd8c5ba2068999a0cd5a26544a83232c0c364.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-27/b57158d5-9d9e-4be4-9032-68fa98ae70ae/97098b9c60595151494e10b6f39cf4267b90f20f52eb4abddc5bf174fc105634.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-27/b57158d5-9d9e-4be4-9032-68fa98ae70ae/7c7e00569e00c3ec72b10bf64f7f3d4c0a084ee64de8c6718a1ea71c8993afdc.jpg)



Figure 6: Comparison of the reference and predicted magnitude of the stress on the lower wall across all the time steps for (A) Oldroyd (B) Linear PTT (C) Giesekus constitutive models. The first column shows the reference values and the second column shows the model’s predictions. In the third column, we highlight the differences between the reference values and the model’s predictions on a logarithmic scale.


consider the stenosis in this work as the flow through stenotic vessels exhibits complex and interesting behavior. It is a challenging yet realistic scenario, as blood can be a viscoelastic fluid. Moreover, modeling flow in stenotic vessels can provide insights into hemodynamic parameters such as shear and wall stress, which can be clinically relevant. The input to the algorithm essentially is the velocity field and the boundary conditions on stress and pressure. For all the results discussed in this section, we represent the velocity components $( u , v )$ using an eight-layer deep, fully connected neural network with 128 neurons per hidden layer. We represent the stress components $\left( \tau ^ { x x } , \tau ^ { x y } , \tau ^ { y y } \right)$ with another eight layers deep, fully connected neural network with 128 neurons per hidden layer. We use a third neural network to represent the pressure (p) with an eight-layer deep, fully connected neural network with 64 neurons per hidden layer. We use fully connected neural networks with a constant number of neurons in each layer as, in our experience, such neural network architecture has worked better than increasing or decreasing neurons in each layer for PINNs. All the networks use weight normalization but do not use batch normalization or dropout. We use the swish function as the activation function for all the networks. The swish activation function returns $x \times S ( x )$ for an input x and is known to match or outperform the ReLU activation function consistently. The sigmoid function $( S ( x ) )$ is defined as $\begin{array} { r } { S ( x ) = \frac { 1 } { 1 + e ^ { - x } } . } \end{array}$ 1+e−x . The ReLU function is mathematically defined as Re $\mathbf { \boldsymbol { \mathrm { . U } } } ( \mathbf { \boldsymbol { x } } ) = m a x ( 0 , x )$ . Future work could explore other architectures, such as convolutional neural networks, which may improve the results presented in this section. 


Table 2: Relative error for flow variables at different noise levels for the Oldroyd-B model


<table><tr><td></td><td>u</td><td>v</td><td>p</td></tr><tr><td>0% Noise</td><td><eq>1.6 \times 10^{-3}</eq></td><td><eq>2.2 \times 10^{-3}</eq></td><td><eq>1.94 \times 10^{-1}</eq></td></tr><tr><td>1% Noise</td><td><eq>1.7 \times 10^{-3}</eq></td><td><eq>2.3 \times 10^{-3}</eq></td><td><eq>2.02 \times 10^{-1}</eq></td></tr><tr><td>5% Noise</td><td><eq>1.7 \times 10^{-3}</eq></td><td><eq>2.4 \times 10^{-3}</eq></td><td><eq>1.87 \times 10^{-1}</eq></td></tr><tr><td>10% Noise</td><td><eq>1.8 \times 10^{-3}</eq></td><td><eq>2.6 \times 10^{-3}</eq></td><td><eq>1.69 \times 10^{-1}</eq></td></tr></table>


Table 3: Relative error for the stress components at different noise levels for the Oldroyd-B model


<table><tr><td></td><td><eq>\tau^{xx}</eq></td><td><eq>\tau^{xy}</eq></td><td><eq>\tau^{yy}</eq></td></tr><tr><td>0% Noise</td><td><eq>5.99 \times 10^{-2}</eq></td><td><eq>2.45 \times 10^{-2}</eq></td><td><eq>2.28 \times 10^{-2}</eq></td></tr><tr><td>1% Noise</td><td><eq>5.93 \times 10^{-2}</eq></td><td><eq>2.48 \times 10^{-2}</eq></td><td><eq>2.31 \times 10^{-2}</eq></td></tr><tr><td>5% Noise</td><td><eq>5.94 \times 10^{-2}</eq></td><td><eq>2.47 \times 10^{-2}</eq></td><td><eq>2.10 \times 10^{-2}</eq></td></tr><tr><td>10% Noise</td><td><eq>6.02 \times 10^{-2}</eq></td><td><eq>2.55 \times 10^{-2}</eq></td><td><eq>2.36 \times 10^{-2}</eq></td></tr></table>


Table 4: Sensitivity to noise level in the velocity data for the Oldroyd model


<table><tr><td></td><td><eq>\alpha</eq></td><td><eq>\epsilon</eq></td><td><eq>\lambda</eq></td><td><eq>\eta_p</eq></td><td><eq>\eta_s</eq></td></tr><tr><td>Reference value</td><td>0.00</td><td>0.0</td><td>0.05</td><td>0.008</td><td>0.01</td></tr><tr><td>0% noise</td><td>0.00</td><td>0.0</td><td>0.0517</td><td>0.0081</td><td>0.0098</td></tr><tr><td>1% noise</td><td>0.00</td><td>0.0</td><td>0.0517</td><td>0.0081</td><td>0.0115</td></tr><tr><td>5% noise</td><td>0.00</td><td>0.0</td><td>0.0517</td><td>0.0081</td><td>0.0115</td></tr><tr><td>10% noise</td><td>0.00</td><td>0.0</td><td>0.0517</td><td>0.0081</td><td>0.0112</td></tr></table>

The learning rate schedule is an important hyperparameter that determines how well the network parameters are optimized. For all the results reported in this work, we use a cosine annealing learning rate schedule [43]. The annealing learning rate schedule starts with a large learning rate, gradually decreasing to the defined minimum value. This allows for exploration while optimizing the parameters, and the reduction in the learning rate value refines the search close to the optima. We used a value of 2.5e-03 for $\zeta _ { m a x }$ and 2.5e-06 for $\zeta _ { m i n }$ to get the learning rate $\zeta$ as defined in the following 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-27/b57158d5-9d9e-4be4-9032-68fa98ae70ae/0de5eaedc6f540d8a07fd7befd34c2d3a0163022acbbf89884a261046b09b81e.jpg)



Figure 7: Comparison of the reference and predicted magnitude of the stress in the throat of the stenosis across all the timesteps for (A) Oldroyd (B) Linear PTT (C) Giesekus constitutive models. The first column shows the reference values, and the second column shows the model’s predictions. In the third column, we highlight the differences between the reference values and the model’s predictions on a logarithmic scale.



Table 5: Relative error for flow variables at different noise levels for the linear PTT model


<table><tr><td></td><td>u</td><td>v</td><td>p</td></tr><tr><td>0% Noise</td><td><eq>1.6 \times 10^{-3}</eq></td><td><eq>2.2 \times 10^{-3}</eq></td><td><eq>1.33 \times 10^{-1}</eq></td></tr><tr><td>1% Noise</td><td><eq>1.8 \times 10^{-3}</eq></td><td><eq>2.4 \times 10^{-3}</eq></td><td><eq>1.69 \times 10^{-1}</eq></td></tr><tr><td>5% Noise</td><td><eq>1.9 \times 10^{-3}</eq></td><td><eq>2.6 \times 10^{-3}</eq></td><td><eq>1.77 \times 10^{-1}</eq></td></tr><tr><td>10% Noise</td><td><eq>2.2 \times 10^{-3}</eq></td><td><eq>3 \times 10^{-3}</eq></td><td><eq>1.85 \times 10^{-1}</eq></td></tr></table>


Table 6: Relative error for the stress components at different noise levels for the linear PTT model


<table><tr><td></td><td><eq>\tau^{xx}</eq></td><td><eq>\tau^{xy}</eq></td><td><eq>\tau^{yy}</eq></td></tr><tr><td>0% Noise</td><td><eq>5.45 \times 10^{-2}</eq></td><td><eq>3.64 \times 10^{-2}</eq></td><td><eq>4.62 \times 10^{-2}</eq></td></tr><tr><td>1% Noise</td><td><eq>5.65 \times 10^{-2}</eq></td><td><eq>3.77 \times 10^{-2}</eq></td><td><eq>4.7 \times 10^{-2}</eq></td></tr><tr><td>5% Noise</td><td><eq>5.66 \times 10^{-2}</eq></td><td><eq>3.82 \times 10^{-2}</eq></td><td><eq>4.75 \times 10^{-2}</eq></td></tr><tr><td>10% Noise</td><td><eq>5.91 \times 10^{-2}</eq></td><td><eq>3.97 \times 10^{-2}</eq></td><td><eq>4.78 \times 10^{-2}</eq></td></tr></table>

equation 

$$
\zeta = \zeta_ {m i n} + 0. 5 (\zeta_ {m a x} - \zeta_ {m i n}) \left(1 + c o s \left(\frac {T _ {c u r}}{T _ {m a x}} \pi\right)\right), \tag {25}
$$

where $T _ { c u r }$ is the current time step and $T _ { m a x }$ is the total time step. For learning the parameters in the general equation, eq. (13), and the stress field, two million iterations of the Adam optimizer [44] were used. As we sequentially solve for the pressure, we first optimize the parameters of the neural network for the stress (θ) by minimizing the loss function defined in eq. (20) and freezing them. We then optimize the parameters for the neural network for velocity (ϕ) and pressure (κ) by minimizing the loss specified in eq. (24). We ran 800,000 iterations of the Adam optimizer for this optimization process with the same learning rate schedule defined above. We investigated different learning rate schedules, such as different values for the maximum and minimum values for the learning rate, while using cosine annealing and a step function to decay the learning rate. It was observed that using the same learning rate for both trainings leads to better results. A description of the loss function used to train the model is shared in 6. We consider the geometry of a 2D stenosis as shown in figure 5 for all the results discussed in this section. While generating the reference dataset, we applied a sinusoidal boundary condition for the inlet velocity. The simulation ran for hundred time steps, or half a sine wave ( 0 to π). As discussed in section 2, we choose a sequential approach to solve for the stress and the pressure. Given the initial and boundary conditions on the stress, we are simultaneously solving the inverse problem of learning the parameters of the general equation, eq. (13), and the forward problem of discovering the stress field in the spatio-temporal domain. To compare the results predicted by the neural networks to the reference value and simulation results, we define the relative error to be 


Table 7: Sensitivity to the noise level in the velocity data for the linear PTT model


<table><tr><td></td><td><eq>\alpha</eq></td><td><eq>\epsilon</eq></td><td><eq>\lambda</eq></td><td><eq>\eta_p</eq></td><td><eq>\eta_s</eq></td></tr><tr><td>Reference value</td><td>0.00</td><td>0.1</td><td>0.15</td><td>0.015</td><td>0.01</td></tr><tr><td>0% noise</td><td>0.00</td><td>0.106</td><td>0.161</td><td>0.0157</td><td>0.0097</td></tr><tr><td>1% noise</td><td>0.00</td><td>0.108</td><td>0.162</td><td>0.0157</td><td>0.0123</td></tr><tr><td>5% noise</td><td>0.00</td><td>0.108</td><td>0.162</td><td>0.0157</td><td>0.0123</td></tr><tr><td>10% noise</td><td>0.00</td><td>0.108</td><td>0.161</td><td>0.0157</td><td>0.0127</td></tr></table>


Table 8: Relative error for flow variables at different noise levels for the Giesekus model


<table><tr><td></td><td>u</td><td>v</td><td>p</td></tr><tr><td>0% Noise</td><td><eq>1.3 \times 10^{-3}</eq></td><td><eq>2.0 \times 10^{-3}</eq></td><td><eq>1.95 \times 10^{-1}</eq></td></tr><tr><td>1% Noise</td><td><eq>1.4 \times 10^{-3}</eq></td><td><eq>2.1 \times 10^{-3}</eq></td><td><eq>1.82 \times 10^{-1}</eq></td></tr><tr><td>5% Noise</td><td><eq>1.4 \times 10^{-3}</eq></td><td><eq>2.2 \times 10^{-3}</eq></td><td><eq>1.83 \times 10^{-1}</eq></td></tr><tr><td>10% Noise</td><td><eq>1.7 \times 10^{-3}</eq></td><td><eq>2.5 \times 10^{-3}</eq></td><td><eq>1.77 \times 10^{-1}</eq></td></tr></table>

$$
\mathcal {L} (a _ {\text { reference }}, a _ {\text { prediction }}) = \sqrt {\frac {(a _ {\text { reference }} - a _ {\text { prediction }}) ^ {2}}{(a _ {\text { reference }} - \bar {a} _ {\text { reference }}) ^ {2}}}, \tag {26}
$$

where the bar denotes the mean value, we use this definition for error so that the multiplication or addition of a constant does not change it. We show the relative error between the predicted and reference values for the stress, velocity, and pressure fields in Fig. 3 and Fig. 4. As expected, the lowest errors are for the velocity fields, as there is data on those fields. The errors are lowest at the initial time steps since the initial condition for the stress is known. The non-monotonic nature of the errors is due to the sinusoidal boundary condition. The agreement between the reference and predicted values is satisfactory as the mean relative error in the stress magnitude is less than 5% for all cases. To test the effect of noise in boundary conditions on the model, we added 1%, 5%, 10%, and 25% Gaussian noise to the Dirichlet boundary condition on stress. Despite intentionally corrupting the boundary data, the model demonstrated resilience by learning an equivalent set of parameters across all noise levels. This insensitivity suggests that the model successfully captures the underlying dynamics instead of overfitting to specifics or noise in the boundary conditions. Since we solve for the pressure field in a decoupled manner, errors in the stress field propagate, resulting in increased errors in the pressure field. Similar observations have been noted in other studies [32]. 

In Fig. 7, we plot the reference and predicted values of the magnitude of the stress in the throat of the stenosis across all the time steps. Although there is an excellent qualitative and quantitative agreement between the predicted and reference values, the model seems to under-predict the magnitude of the stress on the walls. To focus on the stress on the walls, in Fig. 6, we plot the reference and predicted values of the magnitude of stress on the lower wall across all the time steps for the Oldroyd, Linear PTT and Giesekus models. The stress magnitude on the lower and upper walls is symmetric, so we show the results only for the lower wall. The predictions for the Giesekus model perform best, with excellent qualitative and quantitative agreement between the reference and predicted values. However, the model under-predicts the peak value in all cases. 

To check the robustness of our framework, we add Gaussian noise to the velocity observations. The effect of noise on the parameters for the Oldroyd-B, linear PTT, and Giesekus models are reported in the tables 4, 7, and 10, respectively. Adding Gaussian noise does not significantly affect the parameters learned for eq. (13). However, there is an increase in the error for the learned viscosity ηs. Interestingly, the error does not increase as we increase the amount of Gaussian noise. The reported values of ϵ and α illustrate how our framework facilitates model selection. All the learned values align consistently with the conditions specified in Table 1. For the Oldroyd-B model, the learned values for both ϵ and α are equal to zero. In comparison, only the value for ϵ is zero for the Gieskus model, and only α equals zero for the Linear PTT models. If both the learned values of ϵ and α are nonzero, it implies that none of the three constitutive equations can model the fluid, and new constitutive equations need to be considered. The error for the learned velocity, pressure, and stress components for the Oldroyd-B, linear PTT, and Giesekus models are reported in Tables 2, 3, 5, 6, 8, and 9. The general trend is that the error for each variable increases slightly as the noise level increases, but the increase in error is not significant. 

We believe this low sensitivity to Gaussian noise occurs because the model uses many data points. The models were trained on 5.78 million spatio-temporal data points of velocity. We tried training our model on fewer data points to test this hypothesis. Specifically, we consider the Giesekus model with 5.78 million, 578 thousand, 57.8 thousand, and 5.78 thousand data points with 5% Gaussian noise in the velocity data. The results for the parameters are summarized in Table 11. The results start to deteriorate at about 57.8 thousand points, with the value for viscosity $( \eta _ { s } )$ being off by about 50%. The model fails to learn the viscosity (ηs) with 5.78 thousand spatiotemporal points but still learns the parameters of the general equation, eq. 13, reasonably well. We conducted this study to test the feasibility of using our framework with flow visualization techniques such as PIV. While the resolution can vary, about 500 spatial locations per time step is a realistic estimate of the resolution for a PIV experiment. Considering 500 spatial locations over 100 time steps, a realistic number would be getting 50,000 spatiotemporal points from an experiment. 

These results lead us to a promising conclusion that the model performs well with noisy and sparse datasets, a significant advantage considering the often noisy nature of experimental data and the challenges of acquiring high-resolution data. This opens up exciting possibilities for integrating our approach with experimentally acquired datasets. If the velocity field is obtained experimentally, our method can potentially learn the stress field and pressure field and select the appropriate constitutive equation among the discussed models, provided that the boundary conditions are known. This exciting capability paves the way for practical applications in experimental fluid mechanics and constitutive modeling. 


Table 9: Relative error for the stress components at different noise levels for the Giesekus model


<table><tr><td></td><td><eq>\tau^{xx}</eq></td><td><eq>\tau^{xy}</eq></td><td><eq>\tau^{yy}</eq></td></tr><tr><td>0% Noise</td><td><eq>3.88 \times 10^{-2}</eq></td><td><eq>2.16 \times 10^{-2}</eq></td><td><eq>2.88 \times 10^{-2}</eq></td></tr><tr><td>1% Noise</td><td><eq>3.75 \times 10^{-2}</eq></td><td><eq>2.12 \times 10^{-2}</eq></td><td><eq>2.83 \times 10^{-2}</eq></td></tr><tr><td>5% Noise</td><td><eq>3.83 \times 10^{-2}</eq></td><td><eq>2.17 \times 10^{-2}</eq></td><td><eq>2.89 \times 10^{-2}</eq></td></tr><tr><td>10% Noise</td><td><eq>3.85 \times 10^{-2}</eq></td><td><eq>2.19 \times 10^{-2}</eq></td><td><eq>2.89 \times 10^{-2}</eq></td></tr></table>


Table 10: Sensitivity to the noise level in the velocity data for the Gieseukus model


<table><tr><td></td><td><eq>\alpha</eq></td><td><eq>\epsilon</eq></td><td><eq>\lambda</eq></td><td><eq>\eta_p</eq></td><td><eq>\eta_s</eq></td></tr><tr><td>Reference value</td><td>0.2</td><td>0.0</td><td>0.1</td><td>0.01</td><td>0.01</td></tr><tr><td>0% noise</td><td>0.205</td><td>0.0</td><td>0.105</td><td>0.0094</td><td>0.0103</td></tr><tr><td>1% noise</td><td>0.205</td><td>0.0</td><td>0.105</td><td>0.0098</td><td>0.0103</td></tr><tr><td>5% noise</td><td>0.205</td><td>0.0</td><td>0.105</td><td>0.0097</td><td>0.0103</td></tr><tr><td>10% noise</td><td>0.205</td><td>0.0</td><td>0.105</td><td>0.0099</td><td>0.0103</td></tr></table>


Table 11: Sensitivity to amount of spatio-temporal data


<table><tr><td></td><td><eq>\alpha</eq></td><td><eq>\epsilon</eq></td><td><eq>\lambda</eq></td><td><eq>\eta_p</eq></td><td><eq>\eta_s</eq></td></tr><tr><td>Reference value</td><td>0.2</td><td>0.0</td><td>0.1</td><td>0.01</td><td>0.01</td></tr><tr><td>5.78 million</td><td>0.205</td><td>0.0</td><td>0.105</td><td>0.0097</td><td>0.0103</td></tr><tr><td>578 thousand</td><td>0.205</td><td>0.0</td><td>0.105</td><td>0.0103</td><td>0.0093</td></tr><tr><td>57.8 thousand</td><td>0.206</td><td>0.0</td><td>0.106</td><td>0.0103</td><td>0.0155</td></tr><tr><td>5.78 thousand</td><td>0.217</td><td>0.0</td><td>0.111</td><td>0.0104</td><td>0.00045</td></tr></table>

# 3.2 Cross-slot

We now examine a cross-slot geometry, a popular test case for constitutive models of non-Newtonian fluids. We used RheoTool [40] to generate the reference dataset for this problem. As with the previous geometry, the velocity boundary condition at the inlets is transient and varies sinusoidally. The loss function used to train the model is shared in 6. We consider a hundred time steps as our reference dataset. The neural network architecture and input features were the same as those used for the stenosis problem in section 3.1, and we again chose a sequential approach to solve for the stress and then for the pressure. We need two different learning rate schedules for this geometry. We used the cosine annealing learning rate described in eq. (25), but we used different values of $\zeta _ { m a x }$ for the parameters $\alpha , \epsilon , \lambda , \eta _ { p }$ and $\eta _ { s }$ than for the weights and biases. The value of $\zeta _ { m a x }$ for the parameters mentioned above was 2.5e-04, while it was 2.5e-3 for the weights and biases. The $\zeta _ { m i n }$ value was 2.5e-06 for all the parameters. The 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-27/b57158d5-9d9e-4be4-9032-68fa98ae70ae/faaff1c7ef380cbb378d17e346125d2d0d11fb901d79bf9ab7d2aff2c334ce23.jpg)



Figure 8: Comparison of the reference and predicted magnitude of the stress in the cross-slot at the 50th times step for (A) Oldroyd (B) Linear PTT (C) Giesekus constitutive models. The first column shows the reference values, the second column shows the model’s predictions, and the third column shows the difference in the two values on a logarithmic scale.


output features were the velocity field, the stress field, and the pressure field. The boundary conditions for stress and pressure are the same as defined in section 3.1, and the loss functions to solve for the stress and the pressure sequentially remain eqs. (30) and (34), respectively. 


Table 12: Relative error averaged across all time steps for the stress components for the cross-slot geometry for the Oldroyd-B, Linear PTT and Giesekus models.


<table><tr><td></td><td><eq>\tau^{xx}</eq></td><td><eq>\tau^{xy}</eq></td><td><eq>\tau^{yy}</eq></td></tr><tr><td>Oldroyd-B</td><td><eq>1.80 \times 10^{-1}</eq></td><td><eq>2.31 \times 10^{-1}</eq></td><td><eq>1.94 \times 10^{-1}</eq></td></tr><tr><td>Linear PTT</td><td><eq>1.29 \times 10^{-1}</eq></td><td><eq>1.66 \times 10^{-1}</eq></td><td><eq>1.58 \times 10^{-1}</eq></td></tr><tr><td>Giesekus</td><td><eq>1.94 \times 10^{-1}</eq></td><td><eq>2.37 \times 10^{-1}</eq></td><td><eq>2.01 \times 10^{-1}</eq></td></tr></table>


Table 13: Relative error averaged across all time steps for flow variables for the cross-slot geometry for the Oldroyd-B, Linear PTT and Giesekus models.


<table><tr><td></td><td>u</td><td>v</td><td>p</td></tr><tr><td>Oldroyd-B</td><td><eq>2.46 \times 10^{-2}</eq></td><td><eq>2.32 \times 10^{-2}</eq></td><td><eq>1.17 \times 10^{1}</eq></td></tr><tr><td>Linear PTT</td><td><eq>1.39 \times 10^{-2}</eq></td><td><eq>1.23 \times 10^{-2}</eq></td><td><eq>1.17 \times 10^{1}</eq></td></tr><tr><td>Giesekus</td><td><eq>2.29 \times 10^{-2}</eq></td><td><eq>2.24 \times 10^{-2}</eq></td><td><eq>1.19 \times 10^{1}</eq></td></tr></table>

Tables 12 and 13 show the errors for the stress field and the flow variables, respectively. The errors have been computed using the description in eq. (26). Fig. 8 compares the reference and predicted stress magnitudes at the 50th time step. Our model was able to estimate the parameters of eq. (13) with reasonable accuracy, but it had higher errors for the stress field than in the stenosis case. The error in the stress field also affected the accuracy of the pressure field, which depends on the stress field. However, our model captured the viscosity very well, as it accurately reproduces the stress field in most of the domain. The primary source of error was at the corners of the cross-slot, where the stress field had sharp peaks that our model could not capture. This error happened because our model used a single global network, which tended to smooth over these discontinuities, and we could not capture the peak value of stress at the corners. A possible way to overcome this limitation is to use multiple networks or domain decomposition, which can be explored in future work. Table 14 shows the learned parameters of eq. (13). To test the sensitivity of our model to boundary conditions and strain rates, we doubled the flow rate for the second dataset of Linear PTT (#2). It was observed that this increase did not significantly affect the values of the learned parameters. Our framework can be applied to linear and nonlinear regimes if the fluid follows one of the constitutive equations presented in this work. To evaluate the effectiveness of our framework in learning different parameter combinations, we tested it on three distinct parameter sets for the Oldroyd-B model. The framework was able to learn all three different sets of parameters and select the model accurately. Additionally, to assess our framework’s capability for model selection, we tested it using velocity data from the extended Pom-Pom or XPomPom model [?], which is not 

Table 14: The reference and predicted values of the parameters for the Gieseukus, linear PTT, and Oldroyd-B models for the cross-slot geometry. The second dataset (#2) of the linear PTT model considers doubling the flow rate while keeping the same parameters. To evaluate the effect of different parameter combinations on the efficacy of the framework, we examined three cases of the Oldroyd-B model. Additionally, we tested the framework’s capability of model selection by applying it to the flow field obtained from an extended Pom-Pom (XPomPom) constitutive equation. 

<table><tr><td></td><td><eq>\alpha</eq></td><td><eq>\epsilon</eq></td><td><eq>\lambda</eq></td><td><eq>\eta_p</eq></td><td><eq>\eta_s</eq></td></tr><tr><td>Giesekus Reference value</td><td>0.05</td><td>0.0</td><td>0.004</td><td>0.003</td><td>0.01</td></tr><tr><td>Giesekus Predicted value</td><td>0.056</td><td>0.0</td><td>0.00386</td><td>0.0273</td><td>0.011</td></tr><tr><td>Linear PTT Reference value</td><td>0.0</td><td>0.02</td><td>0.008</td><td>0.025</td><td>0.01</td></tr><tr><td>Linear PTT Predicted value</td><td>0.0</td><td>0.0183</td><td>0.0085</td><td>0.0245</td><td>0.0099</td></tr><tr><td>Linear PTT Reference value #2</td><td>0.0</td><td>0.02</td><td>0.008</td><td>0.025</td><td>0.01</td></tr><tr><td>Linear PTT Predicted value #2</td><td>0.0</td><td>0.0228</td><td>0.00836</td><td>0.0239</td><td>0.0102</td></tr><tr><td>Oldroyd-B Reference value</td><td>0.0</td><td>0.0</td><td>0.005</td><td>0.01</td><td>0.01</td></tr><tr><td>Oldroyd-B Predicted value</td><td>0.0</td><td>0.0</td><td>0.0046</td><td>0.0188</td><td>0.011</td></tr><tr><td>Oldroyd-B Reference value #2</td><td>0.0</td><td>0.0</td><td>0.015</td><td>0.01</td><td>0.02</td></tr><tr><td>Oldroyd-B Predicted value #2</td><td>0.0</td><td>0.0</td><td>0.0135</td><td>0.0193</td><td>0.0236</td></tr><tr><td>Oldroyd-B Reference value #3</td><td>0.0</td><td>0.0</td><td>0.01</td><td>0.025</td><td>0.02</td></tr><tr><td>Oldroyd-B Predicted value #3</td><td>0.0</td><td>0.0</td><td>0.0093</td><td>0.033</td><td>0.0171</td></tr><tr><td>XPomPom Predicted value</td><td>37.29</td><td>0.00423</td><td>0.29</td><td>0.145</td><td>0.011</td></tr></table>

included among the three models considered in our study. The learned parameters did not satisfy any of the criteria listed in Table 1 as both ϵ and α had a non-zero value, indicating that none of the three models (Gieseukus, linear PTT, or Oldroyd-B models) was a suitable fit for this flow. 

In this study, we have considered three models and developed a framework to identify which of these three models best fits the data. This represents an important advancement in integrating machine learning and physics-informed neural networks to address challenges in the constitutive modeling of viscoelastic fluids. We have not expanded this work beyond these three constitutive equations since our forward solver has been developed only for these three constitutive equations. If none of these constitutive equations are appropriate for a dataset, the neural network will notify the user by the learned values of ϵ and α. We encourage further developments based on the ideas presented in this paper to include a wider range of constitutive equations. 

# 4 Conclusions and future scope of work

Machine learning algorithms are proving to be an increasingly useful tool in solving problems in fluid mechanics. However, the cost of high-fidelity data often makes utilizing these data-intensive tools impractical. We introduce viscoelasticNet, a physicsinformed neural networks (PINNs)-based framework to address this. This framework selects the viscoelastic constitutive model and learns the stress field from a velocity flow field. We work with three commonly used non-linear viscoelastic models: the Oldroyd-B, Giesekus, and Linear PTT, and build a generalized framework to model them. The velocity, pressure, and stress fields are represented using neural networks. The backward Euler method was used to construct PINNs for the viscoelastic constitutive model. We use a multistage approach to solve the problem by first solving for the stress and then using the stress and velocity fields to solve for the pressure. To test our framework, we used noisy and sparse data sets in this work. We observed that the framework could learn the parameters of the viscoelastic constitutive model reasonably well in all the cases. 

In this work, we applied the viscoelasticNet framework to a stenosis geometry in two dimensions with the above-mentioned constitutive models and also examined the flow in a cross-slot. While our framework could learn the constitutive equation parameters with reasonable accuracy for both cases, it did not capture the peak stress at the corners of the cross-slot well. To address this, we propose exploring a smaller domain instead of a global function. This framework has the potential to be extended to include other rheological constitutive models like the FENE-P and extended Pom-Pom models. We also suggest learning the entire equation instead of just the parameters in a fixed constitutive equation. Future research could consider more complex geometries and three-dimensional cases. The framework we present here can augment techniques like particle image velocimetry (PIV). While PIV can acquire the velocity flow field, our method can acquire the pressure and stress fields. Once the constitutive equation is learned, the parameters can be used to model any future applications of the pertinent fluid. 

# 5 Acknowledgements

A.M.A. acknowledges financial support from the National Science Foundation (NSF) through Grant No. CBET-2141404. 

# 6 Details on training

To enforce the Neumann boundary conditions, we use the normal vectors for the wall $( \pmb { n } ^ { w } = ( l ^ { w } , m ^ { w } ) )$ ) and the outlet $( \pmb { n } ^ { o } = ( l ^ { o } , m ^ { o } ) ,$ ). We enforce the boundary conditions at the wall using 

$$
L _ {w a l l} (\theta) = \mathbb {E} _ {(t ^ {w}, \boldsymbol {x} ^ {w}, \boldsymbol {n} ^ {w})} \left[ | \boldsymbol {\tau} _ {x} (t ^ {w}, \boldsymbol {x} ^ {w}; \theta) l ^ {w} + \boldsymbol {\tau} _ {y} (t ^ {w}, \boldsymbol {x} ^ {w}; \theta) m ^ {w} | ^ {2} \right], \tag {27}
$$

where $( t ^ { w } , \pmb { x } ^ { w } )$ is the spatio-temporal point cloud on the walls of the domain. We enforce the boundary condition at the outlet using 

$$
L _ {o u t l e t} (\theta) = \mathbb {E} _ {(t ^ {o}, \boldsymbol {x} ^ {o}, \boldsymbol {n} ^ {o})} \left[ | \boldsymbol {\tau} _ {x} (t ^ {o}, \boldsymbol {x} ^ {o}; \theta) l ^ {o} + \boldsymbol {\tau} _ {y} (t ^ {o}, \boldsymbol {x} ^ {o}; \theta) m ^ {o} | ^ {2} \right], \tag {28}
$$

where $( t ^ { o } , \pmb { x } ^ { o } )$ is the spatio-temporal point cloud on the outlet of the domain. For the Dirichlet boundary condition at the inlet, we have 

$$
L _ {i n l e t} (\theta) = \mathbb {E} _ {(t ^ {i}, \boldsymbol {x} ^ {i})} \left[ \frac {\left| \boldsymbol {\tau} (t ^ {i} , \boldsymbol {x} ^ {i} ; \theta) - \boldsymbol {\tau} ^ {i} \right| ^ {2}}{\sigma_ {\boldsymbol {\tau}} {} ^ {2}} \right], \tag {29}
$$

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-27/b57158d5-9d9e-4be4-9032-68fa98ae70ae/75a19304b228a8fba3ddd9913f62314bc1b0e1df9075e7ac6545c4b31bf8ed1e.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-27/b57158d5-9d9e-4be4-9032-68fa98ae70ae/89920c01a63c648817dda9828967f56aeefe28b8a7e6fa6a5ea2dcaaddbabfac.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-27/b57158d5-9d9e-4be4-9032-68fa98ae70ae/7b7f38756679d5b89a0597f4db777046e83c3138fcaa320f6468b1c9f83b9bca.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-27/b57158d5-9d9e-4be4-9032-68fa98ae70ae/31dafe933660a14ba8bd29aa9d51ec10449b107d7214b066f4362d28c510def4.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-27/b57158d5-9d9e-4be4-9032-68fa98ae70ae/72497f62644252bcf3549f271933afccd419f1d7e7dec8812f1c05c9bb7c83e3.jpg)



Figure 9: The figure shows the value of the loss terms in eq. (20) as the network parameters are optimized. We observe that the optimizer uniformly reduces all the loss terms.


where $( t ^ { i } , \pmb { x } ^ { i } )$ are the spatiotemporal point cloud at the inlet of the domain, and $\tau ^ { i }$ is the stress field at the inlet. We then use eq. (20) to define our loss function as 

$$
L _ {s t r e s s} (\theta , \phi) = L _ {v e l} (\phi) + L _ {\text {initial}} (\theta) + L _ {w a l l} (\theta) + I _ {s t r e s s} (\theta) + I _ {v e l} (\theta) + I _ {w a l l} (\theta) + (3 0)
$$

$$
L _ {o u t l e t} (\theta) + L _ {i n l e t} (\theta) + L _ {c o n s i s t e n c y} (\theta , \phi).
$$

We enforce the Neumann boundary conditions on the wall for the pressure as 

$$
K _ {w a l l} (\kappa) = \mathbb {E} _ {\left(t ^ {w}, \boldsymbol {x} ^ {w}, \boldsymbol {n} ^ {w}\right)} \left[ \left| p _ {x} \left(t ^ {w}, \boldsymbol {x} ^ {w}; \kappa\right) l ^ {w} + p _ {y} \left(t ^ {w}, \boldsymbol {x} ^ {w}; \kappa\right) m ^ {w} \right| ^ {2} \right], \tag {31}
$$

and to enforce the Neumann boundary conditions, we use the normal vectors for the inlet $( n ^ { i } = ( l ^ { i } , m ^ { i } ) )$ ). For the inlet, we enforce 

$$
K _ {i n l e t} (\kappa) = \mathbb {E} _ {(t ^ {i}, \boldsymbol {x} ^ {i}, \boldsymbol {n} ^ {i})} \left[ | p _ {x} (t ^ {i}, \boldsymbol {x} ^ {i}; \kappa) l ^ {i} + p _ {y} (t ^ {i}, \boldsymbol {x} ^ {i}; \kappa) m ^ {i} | ^ {2} \right]. \tag {32}
$$

For the Dirichlet boundary condition at the outlet for the pressure, we have 

$$
K _ {o u t l e t} (\kappa) = \mathbb {E} _ {(t ^ {o}, \boldsymbol {x} ^ {o}, p ^ {o})} \left[ | p (t ^ {o}, \boldsymbol {x} ^ {o}; \kappa) - p ^ {o} | ^ {2} \right], \tag {33}
$$

where $p ^ { o }$ is the pressure field at the outlet. We do not divide by the standard deviation of the pressure as the pressure is zero at the outlet in our case. We optimize the parameters ϕ and κ using eq. (24) to define the following combined loss 

$$
K _ {p r e s s u r e} (\phi , \kappa) = L _ {v e l} (\phi) + K _ {m o m} (\phi , \kappa) + K _ {w a l l} (\kappa) + K _ {i n l e t} (\kappa) + K _ {o u t l e t} (\kappa). (3 4)
$$

We chose the mini-batch size to be 256 for the spatio-temporal point cloud inside the domain and 64 for all the points on the boundary. Every ten iterations of the Adam optimizer took around 0.45 seconds on a NVIDIA Quadro RTX 8000 GPU.a. We use the following parameters for the Adam optimizer TensorFlow provided: $\beta _ { 1 } = 0 . 9$ , $\beta _ { 2 } = 0 . 9 9 9$ , and $\epsilon _ { 1 } = 1 e - 0 7$ . Here $\beta _ { 1 }$ and $\beta _ { 2 }$ are the exponential decay rates for the first and second momentum estimates, respectively, and $\epsilon _ { 1 }$ is a small constant for numerical stability. The total running time for the inverse problem is around 25 hours; this includes training all the neural networks and learning the parameters. We use the default parameters for the Adam optimizer provided by tensorflow. All the networks use weight normalization but do not use batch normalization or dropout. In fig. 9, we illustrate the reduction of the loss terms in eq. (20) throughout the optimization process. It is evident that the optimizer consistently reduces all the loss terms uniformly. 

# References



[1] G. Li, E. Lauga, A. M. Ardekani, Microswimming in viscoelastic fluids, Journal of Non-Newtonian Fluid Mechanics 297 (April) (2021) 104655. doi:10.1016/j. jnnfm.2021.104655. URL https://doi.org/10.1016/j.jnnfm.2021.104655 





[2] C. K. Tung, C. Lin, B. Harvey, A. G. Fiore, F. Ardon, M. Wu, S. S. Suarez, Fluid viscoelasticity promotes collective swimming of sperm, Scientific Reports 7 (1) (2017) 1–9. doi:10.1038/s41598-017-03341-4. 





[3] G. Li, A. M. Ardekani, Collective Motion of Microorganisms in a Viscoelastic Fluid, Physical Review Letters 117 (11) (2016) 1–5. doi:10.1103/PhysRevLett. 117.118001. 





[4] G. J. Li, A. Karimi, A. M. Ardekani, Effect of solid boundaries on swimming dynamics of microorganisms in a viscoelastic fluid, Rheologica Acta 53 (12) (2014) 911–926. doi:10.1007/s00397-014-0796-9. 





[5] R. Hu, S. Tang, M. Mpelwa, Z. Jiang, S. Feng, Research progress of viscoelastic surfactants for enhanced oil recovery, Energy Exploration and Exploitation 39 (4) (2021) 1324–1348. doi:10.1177/0144598720980209. 





[6] B. Wei, L. Romero-Zer´on, D. Rodrigue, Oil displacement mechanisms of viscoelastic polymers in enhanced oil recovery (EOR): a review, Journal of Petroleum Exploration and Production Technology 4 (2) (2014) 113–121. doi:10.1007/ s13202-013-0087-5. 





[7] M. A. Alves, P. J. Oliveira, F. T. Pinho, Numerical Methods for Viscoelastic Fluid Flows, Annual Review of Fluid Mechanics 53 (2021) 509–541. doi:10. 1146/annurev-fluid-010719-060107. 





[8] J. G. Beijer, J. L. Spoormaker, Solution strategies for FEM analysis with nonlinear viscoelastic polymers, Computers and Structures 80 (14-15) (2002) 1213–1229. doi:10.1016/S0045-7949(02)00089-5. 





[9] P. Areias, K. Matouˇs, Finite element formulation for modeling nonlinear viscoelastic elastomers, Computer Methods in Applied Mechanics and Engineering 197 (51-52) (2008) 4702–4717. doi:10.1016/j.cma.2008.06.015. URL http://dx.doi.org/10.1016/j.cma.2008.06.015 





[10] S. Wang, Z. Su, L. Ying, X. Peng, S. Zhu, F. Liang, D. Feng, D. Liang, I. Technologies, ACCELERATING MAGNETIC RESONANCE IMAGING VIA DEEP LEARNING Paul C . Lauterbur Research Center for Biomedical Imaging , SIAT , CAS , Shenzhen , P . R . China Department of Biomedical Engineering and Department of Electrical Engineering , The State Universit, Isbi 2016 (2016) 514–517. 





[11] S. Min, B. Lee, S. Yoon, Deep learning in bioinformatics, Briefings in bioinformatics 18 (5) (2017) 851–869. doi:10.1093/bib/bbw068. 





[12] A. Voulodimos, N. Doulamis, A. Doulamis, E. Protopapadakis, Deep Learning for Computer Vision: A Brief Review, Computational Intelligence and Neuroscience 2018 (2018). doi:10.1155/2018/7068349. 





[13] A. Esteva, K. Chou, S. Yeung, N. Naik, A. Madani, A. Mottaghi, Y. Liu, E. Topol, J. Dean, R. Socher, Deep learning-enabled medical computer vision, npj Digital Medicine 4 (1) (2021) 1–9. doi:10.1038/s41746-020-00376-2. URL http://dx.doi.org/10.1038/s41746-020-00376-2 





[14] T. Young, D. Hazarika, S. Poria, E. Cambria, Recent Trends in Deep Learning Based Natural Language Processing 1–32. 





[15] A. Torfi, R. A. Shirvani, Y. Keneshloo, N. Tavaf, E. A. Fox, Natural Language Processing Advancements By Deep Learning: A Survey (2020) 1–23. URL http://arxiv.org/abs/2003.01200 





[16] S. L. Brunton, B. R. Noack, P. Koumoutsakos, Machine Learning for Fluid Mechanics, Annual Review of Fluid Mechanics 52 (1) (2020) 477–508. doi: 10.1146/annurev-fluid-010719-060214. 





[17] S. L. Brunton, Applying machine learning to study fluid mechanics, Acta Mechanica Sinica/Lixue Xuebao 37 (12) (2021) 1718–1726. doi:10.1007/ s10409-021-01143-6. URL https://doi.org/10.1007/s10409-021-01143-6 





[18] Z. Y. Wan, P. Vlachas, P. Koumoutsakos, T. Sapsis, Data-assisted reduced-order modeling of extreme events in complex dynamical systems, PLoS ONE 13 (5) (2018) 1–22. doi:10.1371/journal.pone.0197704. 





[19] K. Fukami, K. Fukagata, K. Taira, Assessment of supervised machine learning methods for fluid flows, Theoretical and Computational Fluid Dynamics 34 (4) (2020) 497–519. doi:10.1007/s00162-020-00518-y. URL https://doi.org/10.1007/s00162-020-00518-y 





[20] M. Raissi, P. Perdikaris, G. E. Karniadakis, Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations, Journal of Computational Physics 378 (2019) 686–707. doi:10.1016/j.jcp.2018.10.045. URL https://doi.org/10.1016/j.jcp.2018.10.045 





[21] M. Raissi, Deep hidden physics models: Deep learning of nonlinear partial differential equations, Journal of Machine Learning Research 19 (2018) 1–24. 





[22] E. Haghighat, M. Raissi, A. Moure, H. Gomez, R. Juanes, A physics-informed deep learning framework for inversion and surrogate modeling in solid mechanics, Computer Methods in Applied Mechanics and Engineering 379 (2021) 113741. doi:10.1016/j.cma.2021.113741. URL https://doi.org/10.1016/j.cma.2021.113741 





[23] B. Moseley, A. Markham, T. Nissen-Meyer, Finite Basis Physics-Informed Neural Networks (FBPINNs): a scalable domain decomposition approach for solving differential equations (2021). URL http://arxiv.org/abs/2107.07871 





[24] H. Gao, L. Sun, J. X. Wang, PhyGeoNet: Physics-informed geometry-adaptive convolutional neural networks for solving parameterized steady-state PDEs on irregular domain, Journal of Computational Physics 428 (2021) 110079. doi: 10.1016/j.jcp.2020.110079. URL https://doi.org/10.1016/j.jcp.2020.110079 





[25] Z. Fang, A High-Efficient Hybrid Physics-Informed Neural Networks Based on Convolutional Neural Network, IEEE Transactions on Neural Networks and Learning Systems (2021) 1–13doi:10.1109/TNNLS.2021.3070878. 





[26] R. Zhang, Y. Liu, H. Sun, Physics-informed multi-LSTM networks for metamodeling of nonlinear structures, Computer Methods in Applied Mechanics and Engineering 369 (2020) 113226. doi:10.1016/j.cma.2020.113226. URL https://doi.org/10.1016/j.cma.2020.113226 





[27] Y. A. Yucesan, F. A. Viana, Hybrid physics-informed neural networks for main bearing fatigue prognosis with visual grease inspection, Computers in Industry 125 (2021) 103386. doi:10.1016/j.compind.2020.103386. URL https://doi.org/10.1016/j.compind.2020.103386 





[28] L. Yang, X. Meng, G. E. Karniadakis, B-PINNs: Bayesian physics-informed neural networks for forward and inverse PDE problems with noisy data, Journal of Computational Physics 425 (2021) 109913. doi:10.1016/j.jcp.2020.109913. URL https://doi.org/10.1016/j.jcp.2020.109913 





[29] X. Jin, S. Cai, H. Li, G. E. Karniadakis, NSFnets (Navier-Stokes Flow nets): Physics-informed neural networks for the incompressible Navier-Stokes equations (Hui Li). 





[30] C. J. Arthurs, A. P. King, Active training of physics-informed neural networks to aggregate and interpolate parametric solutions to the Navier-Stokes equations, Journal of Computational Physics 438 (2021) 110364. doi:10.1016/j.jcp.2021. 110364. URL https://doi.org/10.1016/j.jcp.2021.110364 





[31] S. Cuomo, V. Schiano, D. Cola, G. Rozza, M. Raissi, Scientific Machine Learning through Physics-Informed Neural Networks : Where we are and What ’ s next. 





[32] M. Raissi, A. Yazdani, G. E. Karniadakis, Hidden fluid mechanics: Learning velocity and pressure fields from flow visualizations (C) (2020) 1–5. doi:10. 1126/science.aaw4741. URL https://science.sciencemag.org/content/367/6481/1026/tab-pdf 





[33] H. Eivazi, M. Tahani, P. Schlatter, R. Vinuesa, Physics-informed neural networks for solving Reynolds-averaged Navier-Stokes equations, Physics of Fluids 34 (7) (2022). doi:10.1063/5.0095270. 





[34] M. M. Almajid, M. O. Abu-Al-Saud, Prediction of porous media fluid flow using physics informed neural networks, Journal of Petroleum Science and Engineering 208 (PA) (2022) 109205. doi:10.1016/j.petrol.2021.109205. URL https://doi.org/10.1016/j.petrol.2021.109205 





[35] S. Cai, Z. Mao, Z. Wang, M. Yin, G. E. Karniadakis, Physics-informed neural networks (PINNs) for fluid mechanics: a review, Acta Mechanica Sinica/Lixue Xuebao 37 (12) (2021) 1727–1738. doi:10.1007/s10409-021-01148-1. URL https://doi.org/10.1007/s10409-021-01148-1 





[36] M. Mahmoudabadbozchelou, G. E. Karniadakis, S. Jamali, nn-PINNs: Non-Newtonian physics-informed neural networks for complex fluid modeling, Soft Matter 18 (1) (2022) 172–185. doi:10.1039/d1sm01298c. 





[37] C. Limited, On the formulation of rheological equations of state, Proceedings of the Royal Society of London. Series A. Mathematical and Physical Sciences 200 (1063) (1950) 523–541. doi:10.1098/rspa.1950.0035. 





[38] H. Giesekus, A simple constitutive equation for polymer fluids based on the concept of deformation-dependent tensorial mobility, Journal of Non-Newtonian Fluid Mechanics 11 (1-2) (1982) 69–109. doi:10.1016/0377-0257(82)85016-7. 





[39] N. Phan-Thien, A Nonlinear Network Viscoelastic Model, Journal of Rheology 22 (3) (1978) 259–283. doi:10.1122/1.549481. 





[40] F. Pimenta, M. A. Alves, Stabilization of an open-source finite-volume solver for viscoelastic fluid flows, Journal of Non-Newtonian Fluid Mechanics 239 (2017) 85–104. doi:10.1016/j.jnnfm.2016.12.002. URL http://dx.doi.org/10.1016/j.jnnfm.2016.12.002 





[41] H. G. Weller, G. Tabor, H. Jasak, C. Fureby, A tensorial approach to computational continuum mechanics using object-oriented techniques, Computers in Physics 12 (6) (1998) 620. doi:10.1063/1.168744. 





[42] J. L. Favero, A. R. Secchi, N. S. Cardozo, H. Jasak, Viscoelastic flow analysis using the software OpenFOAM and differential constitutive equations, Journal of Non-Newtonian Fluid Mechanics 165 (23-24) (2010) 1625–1636. doi:10.1016/j. jnnfm.2010.08.010. URL http://dx.doi.org/10.1016/j.jnnfm.2010.08.010 





[43] I. Loshchilov, F. Hutter, SGDR: Stochastic gradient descent with warm restarts, 5th International Conference on Learning Representations, ICLR 2017 - Conference Track Proceedings (2017) 1–16. 





[44] D. P. Kingma, J. L. Ba, Adam: A method for stochastic optimization, 3rd International Conference on Learning Representations, ICLR 2015 - Conference Track Proceedings (2015) 1–15. 

