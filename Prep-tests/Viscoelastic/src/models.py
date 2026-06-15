import torch
import torch.nn as nn
import numpy as np

class FourierFeatures(nn.Module):
    def __init__(self, in_dim=2, mapping_size=64, sigma=10.0):
        super().__init__()
        B = torch.randn(mapping_size, in_dim) * sigma
        self.register_buffer("B", B)

    def forward(self, x):
        x_proj = 2 * np.pi * x @ self.B.T
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class FCN(nn.Module):
    """Rete Neurale a Connessioni Complete (Fully Connected Network) con supporto RFF"""
    def __init__(self, layers, activation_fn=nn.SiLU, use_rff=True, rff_mapping_size=64, rff_sigma=10.0):
        super().__init__()
        self.use_rff = use_rff
        self.activation = activation_fn()
        
        if self.use_rff:
            self.encoder = FourierFeatures(in_dim=layers[0], mapping_size=rff_mapping_size, sigma=rff_sigma)
            in_dim = rff_mapping_size * 2
        else:
            self.encoder = nn.Identity()
            in_dim = layers[0]
            
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(in_dim, layers[1]))
        for i in range(1, len(layers) - 1):
            self.fcs.append(nn.Linear(layers[i], layers[i+1]))
            
    def forward(self, x):
        x = self.encoder(x)
        for layer in self.fcs[:-1]:
            x = self.activation(layer(x))
        return self.fcs[-1](x) 
    def loss_fn(self, pred, target):
        return nn.MSELoss()(pred, target)

def get_activation_name(activation_class):
    return activation_class.__name__

def format_layers_name(layers):
    if len(layers) > 3:
        hidden = layers[1:-1]
        if all(x == hidden[0] for x in hidden):
            return f"{layers[0]}_{hidden[0]}x{len(hidden)}_{layers[-1]}"
    return "_".join(map(str, layers))

class ViscoelasticCombinedModel(nn.Module):
    def __init__(self, model_psi, model_p, model_tau):
        super().__init__()
        self.model_psi = model_psi
        self.model_p = model_p
        self.model_tau = model_tau
    def forward(self, x):
        psi = self.model_psi(x)
        p = self.model_p(x)
        tau = self.model_tau(x)
        return torch.cat([psi, p, tau], dim=1)

class ScaledViscoelasticCombinedModel(nn.Module):
    """Combined model with output scaling for p and tau to match physical magnitudes."""
    def __init__(self, model_psi, model_p, model_tau, p_scale=1.0, tau_scale=1.0):
        super().__init__()
        self.model_psi = model_psi
        self.model_p = model_p
        self.model_tau = model_tau
        self.p_scale = p_scale
        self.tau_scale = tau_scale

    def forward(self, x):
        psi = self.model_psi(x)
        p = self.model_p(x) * self.p_scale
        tau = self.model_tau(x) * self.tau_scale
        return torch.cat([psi, p, tau], dim=1)

def initialize_last_layer_zero(model):
    last_layer = list(model.fcs)[-1]
    nn.init.zeros_(last_layer.weight)
    nn.init.zeros_(last_layer.bias)
    print(f"  [Init] Ultimo layer di {model.__class__.__name__} inizializzato a zero.")
