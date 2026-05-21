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
In the context of the `PINN-tesi` repository, this technique is suggested for Goal 2 (`SoloData`). 

### Proposed Extension
1. **Target Extension**: Expand the input data structure for `SoloData` to include the exact velocity components $u$ and $v$ in addition to $\psi, p, \tau_{xx}, \tau_{xy}, \tau_{yy}$, moving from a 5-channel target to a 7-channel target:
   `[psi_exact, p_exact, txx_exact, txy_exact, tyy_exact, u_exact, v_exact]`
2. **Loss Calculation (`compute_pinn_loss`)**: In `func/history_tracker.py`, if the target has 7 channels, extract the exact velocity data and compute the derivative loss:
   ```python
   # Compute predicted velocities from psi
   u_pred = torch.autograd.grad(psi_pred.sum(), coords, create_graph=True)[0][..., 1:2]
   v_pred = -torch.autograd.grad(psi_pred.sum(), coords, create_graph=True)[0][..., 0:1]
   
   # Add Sobolev terms
   loss_u_sobolev = torch.mean((u_pred - u_exact) ** 2)
   loss_v_sobolev = torch.mean((v_pred - v_exact) ** 2)
   
   loss_data = loss_data + gamma * (loss_u_sobolev + loss_v_sobolev)
   ```
This avoids altering the model architecture itself, but adds explicit supervision on the derivatives of `model_psi`.

## References & Back-links
- [[ViscoelasticNet]]
- [[Loss_Functions]]
- [[Viscoelastic_Training]]
