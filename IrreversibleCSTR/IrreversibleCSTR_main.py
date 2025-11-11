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



step=100

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

# slice out a small number of points from the LHS of the domain
x_data = x[0:150:10]
y_data = y[0:150:10]
print(x_data.shape, y_data.shape)

plt.figure()
plt.plot(x, y, label="Exact solution")
plt.scatter(x_data, y_data, color="tab:orange", label="Training data")
plt.legend()
plt.show()

    
# train standard neural network to fit training data
torch.manual_seed(123)
model = FCN(1,1,32,3)
optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
files = []
for i in tqdm(range(step), desc="Training NN"):
    optimizer.zero_grad()
    yh = model(x_data)
    loss = torch.mean((yh-y_data)**2)# use mean squared error
    loss.backward()
    optimizer.step()
    
    
    # plot the result as training progresses
    if (i+1) % 100 == 0: 
        
        yh = model(x).detach()
        
        plot_result(i,x,y,x_data,y_data,yh)
        
        file = "plots/CSTRnn_%.8i.png"%(i+1)
        plt.savefig(file, bbox_inches='tight', pad_inches=0.1, dpi=100, facecolor="white")
        files.append(file)
    
        if (i+1) % 5000 == 0: plt.show()# cambiato per non vedere sempre
        else: plt.close("all")
            
save_gif_PIL("IrreversibleCSTR/CSTRnn.gif", files, fps=20, loop=0,delete_files=True)




x_physics = torch.linspace(0,5,50).view(-1,1).requires_grad_(True)# sample locations over the problem domain


torch.manual_seed(123)
model = FCN(1,1,32,3)
optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
files = []
for i in tqdm(range(step), desc="Training PINN"):
    optimizer.zero_grad()
    
    # compute the "data loss"
    yh = model(x_data)
    loss1 = torch.mean((yh-y_data)**2)# use mean squared error
    
    # compute the "physics loss"
    yhp = model(x_physics)
    dx  = torch.autograd.grad(yhp, x_physics, torch.ones_like(yhp), create_graph=True)[0]# computes dy/dx
    physics = dx + k*yhp - F*(cAin-yhp)/V# computes the residual of the CSTR irreversible mass balance equation
    loss2 = (1e-4)*torch.mean(physics**2)
    # penso che qua sotto yhp è cA(t), dx sarebbe dcA/dt
    
    # backpropagate joint loss
    loss = loss1 + loss2# add two loss terms together
    loss.backward()
    optimizer.step()
    
    
    # plot the result as training progresses
    if (i+1) % 100 == 0: 
        
        yh = model(x).detach()
        xp = x_physics.detach()
        
        plot_result(i,x,y,x_data,y_data,yh,xp)
        
        file = "plots/CSTRpinn_%.8i.png"%(i+1)
        plt.savefig(file, bbox_inches='tight', pad_inches=0.1, dpi=100, facecolor="white")
        files.append(file)
        
        if (i+1) % 4500 == 0: plt.show()
        else: plt.close("all")
            
save_gif_PIL("IrreversibleCSTR/CSTRpinn.gif", files, fps=20, loop=0)