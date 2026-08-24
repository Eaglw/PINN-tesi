# Sobolev Regularization

## Overview
Sobolev Regularization (also known as gradient supervision or Sobolev training) is a regularization technique where not only the function values, but also the derivatives of the function, are supervised during training.

In Physics-Informed Neural Networks (PINNs), this is particularly relevant when training with data. For example, in a stream function formulation where velocity components are derived from a stream function $\psi$:
$$u = \frac{\partial \psi}{\partial y}, \quad v = -\frac{\partial \psi}{\partial x}$$
Supervising only $\psi$ (standard $L^2$ training, as in Goal 2 `SoloData`) can lead to high-frequency oscillations in $\psi$ as training progresses (overfitting). While the error in $\psi$ remains very small, these oscillations are amplified during differentiation, leading to severe degradation in the velocity fields $u$ and $v$ over long-epoch training.

By penalizing the difference between the predicted derivatives and the exact derivative data:
$$\mathcal{L}_{\text{Sobolev}} = \mathcal{L}_{\psi} + \gamma_u \mathcal{L}_{u} + \gamma_v \mathcal{L}_{v}$$
where:
$$\mathcal{L}_{u} = \frac{1}{N} \sum_{i=1}^N \left| \frac{\partial \psi}{\partial y} - u_{\text{exact}} \right|^2, \quad \mathcal{L}_{v} = \frac{1}{N} \sum_{i=1}^N \left| -\frac{\partial \psi}{\partial x} - v_{\text{exact}} \right|^2$$
the neural network is constrained to learn a smooth representation of the stream function, preventing velocity degradation over long-epoch training.

## Technical Implementation & Physical Details
In the production framework (`final_roll`), Sobolev derivative supervision is the fundamental mechanism through which the stream function network `model_psi` is trained on internal flow data:

### Pure Velocity Supervision (Semi-Inverse Benchmark)
Per foundational project design rules, **no internal stress or pressure data from CFD is ever fed to the PINN** in the interior domain:
1. **Target Structure**: The observed internal dataset contains strictly $(x, y, u_{\text{obs}}, v_{\text{obs}})$.
2. **Derivative Supervision**: The stream function network outputs a scalar $\psi(x, y)$, and the autograd derivatives:
   $$u_{\text{pred}} = \frac{\partial \psi}{\partial y}, \quad v_{\text{pred}} = -\frac{\partial \psi}{\partial x}$$
   are directly supervised against $(u_{\text{obs}}, v_{\text{obs}})$ via MSE:
   ```python
   # Autograd velocity derivation from psi
   psi_grad = torch.autograd.grad(psi.sum(), coords, create_graph=True)[0]
   u_pred = psi_grad[:, 1:2]
   v_pred = -psi_grad[:, 0:1]

   # Data loss purely in Sobolev/derivative space
   loss_data = torch.mean((u_pred - u_obs) ** 2) + torch.mean((v_pred - v_obs) ** 2)
   ```
This guarantees that mass conservation ($\nabla \cdot \mathbf{u} = 0$) is satisfied exactly while avoiding artificial high-frequency oscillations in the reconstructed stream function.

## References & Back-links
- [[ViscoelasticNet]]
- [[Loss_Functions]]
- [[Viscoelastic_Training]]

