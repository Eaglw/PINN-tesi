from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from func.graphic_func import save_gif_PIL, plot_result

# Configurazione dispositivo e precisione
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)
print(f"Using device: {device} with default dtype: {torch.get_default_dtype()}")

# Imposta il device di default (opzionale, ma utile se gli script chiamati non gestiscono esplicitamente il device)
try:
    torch.set_default_device(device)
except AttributeError:
    pass # Versioni vecchie di pytorch

step=6000
 #step di training condivisi tra i try per comparare 
"""
Seleziona quali casi eseguire inserendo nell'array goal il corrispettivo numero
0. NN classica e PINN con dati e fisica
1. Solo fisica e BC
2. Problema inverso
3. PINN che confronta l'andamento di diversi optimizer e activation function
4. PINN accoppiata (Mass + Energy) ispirata a ViscoelasticNet
"""
goal = [4]

#parametri fisici del problema
F, V, cAin, k, cA0 = 400, 2000, 10, 1, 10


class FCN(nn.Module):
    "Defines a connected network"
    
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super().__init__()
        activation = nn.Tanh
        self.fcs = nn.Sequential(*[
                        nn.Linear(N_INPUT, N_HIDDEN),
                        activation()])
        self.fch = nn.Sequential(*[
                        nn.Sequential(*[
                            nn.Linear(N_HIDDEN, N_HIDDEN),
                            activation()]) for _ in range(N_LAYERS-1)])
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)
        
    def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x

def CSTR(F, V, cAin, k, cA0,  x):
    """Defines the analytical solution to CSTR irreversible reaction"""
    den = (F+V*k)
    exp = torch.exp(-(x*den/V))
    A=(F*cA0 -F*cAin +V*cA0*k)
    B = (F*cAin)
    y  = (exp*A+B)/den
    return y


# get the analytical solution over the full domain
x = torch.linspace(0,5,500).view(-1,1)
y = CSTR(F, V, cAin, k, cA0,  x).view(-1,1)
print(x.shape, y.shape)

# slice out and plot a small number of points from the LHS of the domain
x_data = x[0:150:10]
y_data = y[0:150:10]
print(x_data.shape, y_data.shape)
plt.figure()
plt.plot(x.cpu(), y.cpu(), label="Exact solution")
plt.scatter(x_data.cpu(), y_data.cpu(), color="tab:orange", label="Training data")
plt.legend()
#plt.show()



if 0 in goal:
    print("0. NN classica e PINN con dati e fisica")
    exec(open("IrreversibleCSTR/IrreversibleCSTR_nn_pinn.py").read())
if 1 in goal:
    print("1. Solo fisica e BC")
    exec(open("IrreversibleCSTR/IrreversibleCSTR_nodata.py").read())
if 2 in goal:
    print("2. Problema inverso")
    exec(open("IrreversibleCSTR/IrreversibleCSTR_inverse.py").read())
if 3 in goal:
    print("3. Analisi ottimizzatori e funzioni di attivazione")
    exec(open("IrreversibleCSTR/IrreversibleCSTR_pinn_optim.py").read())
if 4 in goal:
    print("4. PINN accoppiata (Mass + Energy)")
    exec(open("IrreversibleCSTR/IrreversibleCSTR_coupled.py").read())