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

step=20000

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


F, V, cAin, k, cA0 = 400, 2000, 10, 1, 10

# get the analytical solution over the full domain
x = torch.linspace(0,5,500).view(-1,1)
y = CSTR(F, V, cAin, k, cA0,  x).view(-1,1)
print(x.shape, y.shape)

# slice out and plot a small number of points from the LHS of the domain
x_data = x[0:150:10]
y_data = y[0:150:10]
print(x_data.shape, y_data.shape)

plt.figure()
plt.plot(x, y, label="Exact solution")
plt.scatter(x_data, y_data, color="tab:orange", label="Training data")
plt.legend()
plt.show()

if False:
    exec(open("IrreversibleCSTR/IrreversibleCSTR_nn_pinn.py").read())
if True:
    exec(open("IrreversibleCSTR/IrreversibleCSTR_nodata.py").read())