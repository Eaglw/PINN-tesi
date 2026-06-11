import torch
from dataclasses import dataclass, field

@dataclass
class TrainingConfig:
    epochs: int = 1000
    base_lr: float = 1e-3
    adam_eps: float = 1e-7
    lr_strategy: str = 'cosine'
    staged_training: bool = True
    warmup_ratio: float = 0.4
    precision_mode: str = 'staged'
    max_lbfgs_iters: int = 100
    grad_clip_norm: float = 5.0
    param_clip_norm: float = 1.0
    param_lr_factor: float = 0.1
    minibatch_internal: int = 1024
    minibatch_boundary: int = 256
    dynamic_weighting: bool = True
    update_weights_every: int = 100
    loss_weights: dict = field(default_factory=lambda: {'data': 1.0, 'bc': 1.0, 'physics': 1.0})
    group_weights: dict = field(default_factory=lambda: {'Inlet': 1.0, 'Walls': 1.0, 'Outlet': 1.0})
    mode: str = 'standard'
    variance_weights: dict = None
    log_gradients_every: int = 500
    plot_every: int = 500
    experiment_name: str = "VE Training"
    val_label: str = "Value"
    physics_warmup_epochs: int = 0


def set_model_trainable(model_combined, active_components=['psi', 'p', 'tau']):
    for p in model_combined.parameters():
        p.requires_grad = False
    if 'psi' in active_components:
        for p in model_combined.model_psi.parameters(): p.requires_grad = True
    if 'p' in active_components:
        for p in model_combined.model_p.parameters(): p.requires_grad = True
    if 'tau' in active_components:
        for p in model_combined.model_tau.parameters(): p.requires_grad = True
        
    print(f"  [Trainable status] Psi: {'psi' in active_components}, P: {'p' in active_components}, Tau: {'tau' in active_components}")

def set_physics_trainable(physics_problem, active_params=['mu_s', 'mu_p', 'lam', 'eps', 'alpha']):
    if not getattr(physics_problem, 'inverse_mode', False):
        return
    for p_name in ['mu_s', 'mu_p', 'lam', 'eps', 'alpha']:
        p_val = getattr(physics_problem, p_name)
        if isinstance(p_val, torch.Tensor) and p_val.is_leaf:
            p_val.requires_grad_(p_name in active_params)
    print(f"  [Physics Trainable] {active_params}")

def _get_scheduler(optimizer, strategy, total_steps):
    if strategy == 'step_decay':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(total_steps * 0.25), gamma=0.5)
    elif strategy == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=600, min_lr=1e-6, cooldown=3000)
    elif strategy == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    return None
