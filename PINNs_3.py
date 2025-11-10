"""
THIS IS AN EXAMPLE THAT I AM CREATING FROM SCRATCH TO SEE IF I UNDERSTOOD CORRECTLY THE USE OF THE FRAMEWORK
29/01/2025

Working on pytorch. Case-study:

DAMPED HARMONIC OSCILLATOR

https://github.com/benmoseley/harmonic-oscillator-pinn/blob/main/Harmonic%20oscillator%20PINN.ipynb
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def save_gif_PIL(outfile, files, fps=5, loop=0):
    "Helper function for saving GIFs"
    imgs = [Image.open(file) for file in files]
    #imgs = [img.resize((925, 328)) if imgs[0].size[0] != 925 else img for img in imgs]
    imgs[0].save(fp=outfile, format='GIF', append_images=imgs[1:], save_all=True, duration=int(1000 / fps), loop=loop)


def oscillator(d, w0, x):
    """
    Defines the analytical solution to the 1D underdamped harmonic oscillator problem.
    """
    assert d < w0
    w = np.sqrt(w0 ** 2 - d ** 2)
    phi = np.arctan(-d / w)
    A = 1 / (2 * np.cos(phi))
    cos = torch.cos(phi + w * x)
    sin = torch.sin(phi + w * x)
    exp = torch.exp(-d * x)
    y = exp * 2 * A * cos
    return y


# andrebbe testata la gelu invece della tanh

class FCN(nn.Module):
    "Defines a fully connected network"

    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS, ACTIVATION):
        super().__init__()
        activation = ACTIVATION
        self.fcs = nn.Sequential(*[
            nn.Linear(N_INPUT, N_HIDDEN),
            #nn.LayerNorm(N_HIDDEN, elementwise_affine=False),  # adding layer normalization to avoid gradient stagnation in the case of lbfgs
            activation()])
        self.fch = nn.Sequential(*[
            nn.Sequential(*[
                nn.Linear(N_HIDDEN, N_HIDDEN),
                #nn.LayerNorm(N_HIDDEN, elementwise_affine=False),
                activation()]) for _ in range(N_LAYERS - 1)])
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)

    def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x


def reinitialize_weights(model):
    for layer in model.children():
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)  # Xavier/Glorot initialization
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

# here I am declaring what I want to do

objectives = ['nn_train', 'pinn_direct', 'pinn_inverse', 'pinn_inverse_continue', 'pinn_inverse_lbfgs', 'pinn_inverse_better']
goal = objectives[0]

# generating training data
d = 2
w = 20
x = torch.linspace(0, 1, 500).view(-1, 1)
y = oscillator(d, w, x).view(-1, 1)

x_data = x[0:500:20]#x[0:200:1]  # prima era ogni 20
y_data = y[0:500:20]#y[0:200:1]

plt.figure()
plt.plot(x, y, label="Exact solution")
plt.scatter(x_data, y_data, color="tab:orange", label="Training data")
plt.legend()


#plt.show()


def plot_result(x, y, x_data, y_data, yh, xp=None):
    "Pretty plot training results"
    plt.figure(figsize=(8, 4))
    plt.plot(x, y, color="grey", linewidth=2, alpha=0.8, label="Exact solution")
    plt.plot(x, yh, color="tab:blue", linewidth=4, alpha=0.8, label="Neural network prediction")
    plt.scatter(x_data, y_data, s=60, color="tab:orange", alpha=0.4, label='Training data')
    if xp is not None:
        plt.scatter(xp, -0 * torch.ones_like(xp), s=60, color="tab:green", alpha=0.4,
                    label='Physics loss training locations')
    l = plt.legend(loc=(1.01, 0.34), frameon=False, fontsize="large")
    plt.setp(l.get_texts(), color="k")
    plt.xlim(-0.05, 1.05)
    plt.ylim(-1.1, 1.1)
    plt.text(1.065, 0.7, "Training step: %i" % (i + 1), fontsize="xx-large", color="k")
    plt.axis("off")


#save_gif_PIL("pinn_inverse.gif", ['plots/'+element for element in os.listdir(r'plots') if 'pinn_inverse' in element][::10], fps=20, loop=0)

a = 0

if goal == objectives[0]:
    # train standard neural network to fit training data
    torch.manual_seed(123)
    model = FCN(1, 1, 32, 3, nn.Tanh)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    files = []
    for i in range(5000):
        optimizer.zero_grad()
        yh = model(x_data)
        loss = torch.mean((yh - y_data) ** 2)  # use mean squared error
        loss.backward()
        optimizer.step()

        # plot the result as training progresses
        if (i + 1) % 10 == 0:

            yh = model(x).detach()

            plot_result(x, y, x_data, y_data, yh)

            file = "plots/nn_entire_domain_longer%.8i.png" % (i + 1)
            plt.savefig(file, bbox_inches='tight', pad_inches=0.1, dpi=100, facecolor="white")
            files.append(file)

            if (i + 1) % 500 == 0:
                plt.show()
            else:
                plt.close("all")

    #save_gif_PIL("nn_entire_domain_longer.gif", files, fps=20, loop=0)
    exit()

a = 0
########################################################################################################################
# HERE WE ARE SOLVING THE DIRECT PROBLEM

if goal == objectives[1]:
    """
    in this case I want to solve the problem without training data, but using only the equation
    and the boundary condition
    """

    x_physics = torch.linspace(0, 1, 500).view(-1, 1).requires_grad_(True)  # sample locations over the problem domain
    mu, k = 2 * d, w ** 2

    torch.manual_seed(123)
    model = FCN(1, 1, 32, 3, nn.Tanh)  # input, output, neurons hidden, n hidden layers
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    files = []
    for i in range(30000):  # iterating to learn the function
        optimizer.zero_grad()

        # compute the "data loss" ---> questo non è necessario tecnicamente
        # yh = model(x_data)
        # loss1 = torch.mean((yh - y_data) ** 2)  # use mean squared error

        # compute the "physics loss"
        yhp = model(x_physics)  # questa è la soluzione approssimata nei colocation points (NO SOLUZIONE)
        # torch.autograd(output,input,grad_output,create_graph)
        # output: ---> funzione calcolata nei colocation points
        # input: colocation points( they are NOT used to accumulate gradient )
        # grad_output: usually is a matrix of ones of the same dimension of ouputs
        # create_graph:graph of the derivative will be constructed, allowing to compute higher order derivative products
        dx = torch.autograd.grad(yhp, x_physics, torch.ones_like(yhp), create_graph=True)[0]  # computes dy/dx
        dx2 = torch.autograd.grad(dx, x_physics, torch.ones_like(dx), create_graph=True)[0]  # computes d^2y/dx^2
        physics = dx2 + mu * dx + k * yhp  # computes the residual of the 1D harmonic oscillator differential equation
        # the author most probably used a weight to penalize this loss
        loss2 = torch.mean(physics ** 2) * 1e-3

        # now I need to add boundary and initial conditions (in some points)
        # since it is  1d problem I just need 1 colocation points (at time t=0)!
        single_point = x[0].unsqueeze(1)
        # questo step a quanto pare è necessario!
        single_point.requires_grad = True
        bc1 = model(single_point)

        bc2 = torch.autograd.grad(bc1, single_point, torch.ones_like(bc1), create_graph=True)[0]
        # questa loss limita il training ----> non sta imparando questa condizione!
        loss3 = torch.mean((bc1 - 1) ** 2)
        loss4 = torch.mean(bc2 ** 2)

        # PER ORA HO UN PROBLEMA DI MAGNITUDINE --> la rete sembra cogliere l'andamento ma non la grandezza dell'oscillazione

        # backpropagate joint loss
        loss = loss2 + loss3 + loss4  # + loss1  # add two loss terms together
        loss.backward()
        optimizer.step()

        print("Iteration: {}  | Loss: {}  | Physics Loss: {} | Dirichlet Loss: {} | Neumann loss: {}".format(i, loss,
                                                                                                             loss2,
                                                                                                             loss3,
                                                                                                             loss4))

        # plot the result as training progresses
        if (i + 1) % 150 == 0:

            yh = model(x).detach()
            xp = x_physics.detach()

            plot_result(x, y, x_data, y_data, yh, xp)

            file = "plots/pinn_no_data_%.8i.png" % (i + 1)
            plt.savefig(file, bbox_inches='tight', pad_inches=0.1, dpi=100, facecolor="white")
            files.append(file)

            if (i + 1) % 6000 == 0:
                plt.show()
            else:
                plt.close("all")

    #save_gif_PIL("pinn_no_data.gif", files, fps=20, loop=0)
    exit()

"""
https://towardsdatascience.com/inverse-physics-informed-neural-net-3b636efeb37e/
"""

if goal == objectives[2]:
    torch.manual_seed(123)

    # I am picking data on the entire domain
    x_data_new = x[::2]
    y_data_new = y[::2]

    model = FCN(1, 1, 64, 5, nn.Tanh)

    x_physics = torch.linspace(0, 1, 500).view(-1, 1).requires_grad_(True)

    mu_train = nn.Parameter(torch.tensor(1.0, dtype=torch.float32, requires_grad=True))
    k_train = nn.Parameter(torch.tensor(300.0, dtype=torch.float32, requires_grad=True))

    #boundary conditions
    single_point = x[0].unsqueeze(1)
    single_point.requires_grad = True

    # Optimizer for pretraining (only updates model parameters)
    optimizer_model = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Lists to store the parameter updates
    mu_history = []
    k_history = []
    files = []
    # Pretraining loop
    for epoch in range(6000):
        optimizer_model.zero_grad()

        # Compute losses (solution loss, boundary loss, etc.)
        loss1 = torch.mean((model(x_data_new) - y_data_new) ** 2)  # Data loss
        bc1 = model(single_point)
        bc2 = torch.autograd.grad(bc1, single_point, torch.ones_like(bc1), create_graph=True)[0]
        loss3 = torch.mean((bc1 - 1) ** 2)  # Boundary conditions (if any)
        loss4 = torch.mean(bc2 ** 2)

        loss_pretrain = loss1 + loss3 + loss4  # Pretraining loss

        loss_pretrain.backward()
        optimizer_model.step()

        if epoch % 100 == 0:
            print(f"Pretrain Epoch {epoch}, Loss: {loss_pretrain.item()}")

    print("Pretraining finished. Now optimizing parameters...")

    optimizer_full = torch.optim.Adam([
        {'params': model.parameters(), 'lr': 1e-4},
        {'params': [mu_train, k_train], 'lr': 1e-2}  # Higher learning rate for parameters
    ])

    for i in range(10000000):
        optimizer_full.zero_grad()

        # Solution loss (already trained)
        loss1 = torch.mean((model(x_data_new) - y_data_new) ** 2)
        bc1 = model(single_point)
        bc2 = torch.autograd.grad(bc1, single_point, torch.ones_like(bc1), create_graph=True)[0]
        loss3 = torch.mean((bc1 - 1) ** 2)  # Boundary conditions (if any)
        loss4 = torch.mean(bc2 ** 2)

        # Compute physics loss
        yhp = model(x_physics)  # Predicted solution at physics points
        dx = torch.autograd.grad(yhp, x_physics, torch.ones_like(yhp), create_graph=True, retain_graph=True)[0]
        dx2 = torch.autograd.grad(dx, x_physics, torch.ones_like(dx), create_graph=True, retain_graph=True)[0]
        physics = dx2 + mu_train * dx + k_train * yhp
        loss2 = torch.mean(physics ** 2)  # Physics loss

        # the model tends to go into an overdumped state, therefore I am applying a soft constraint by regularization
        # this approach do not avoid entirely the overdamped state, because it strongly depends on the lambda_reg
        damping_constraint = torch.relu(mu_train - 2 * torch.sqrt(k_train)) ** 2
        lambda_reg = 0.1


        """
        plt.figure()
        plt.plot(x, y, label="Exact solution")
        plt.scatter(x_data_new, y_data_new, color="tab:orange", label="Training data")
        plt.scatter(x_data_new, model(x_data_new).detach().numpy(), color='red', label="Prediction")
        plt.legend()
        
        
        plt.show()
        """


        # Total loss
        loss = loss1 + loss2 + loss3 + loss4 + lambda_reg * damping_constraint

        loss.backward()
        optimizer_full.step()

        # Store the values of mu and k
        mu_history.append(mu_train.item())
        k_history.append(k_train.item())

        if i % 100 == 0:
            print(f"Epoch {i}, Loss: {loss.item()}, mu_train: {mu_train.item()}, k_train: {k_train.item()}"
                  f", Solution Loss: {loss1.item()}, PDE Loss: {loss2.item()}, Dirichlet Loss: {loss3.item()}, Neumann Loss: {loss4.item()}"
                  f", Damping Constraint: {damping_constraint.item()}")

        # plot the result as training progresses
        if (i + 1) % 2000 == 0:

            yh = model(x).detach()
            xp = x_physics.detach()

            plot_result(x, y, x_data_new, y_data_new, yh)

            file = "plots/pinn_inverse_%.8i.png" % (i + 1)
            plt.savefig(file, bbox_inches='tight', pad_inches=0.1, dpi=100, facecolor="white")
            files.append(file)

            if (i + 1) % 6000 == 0:
                plt.show()
            else:
                plt.close("all")




    #save_gif_PIL("pinn_inverse.gif", files, fps=20, loop=0)

    '''
    torch.save({
        'iteration': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer_full.state_dict(),
        'loss': loss,
        'data_loss': loss1,
        'physics_loss': loss2,
        'dirichlet_loss': loss3,
        'numann_loss': loss4,
        'mu': mu_train,
        'k': k_train,
        'mu_history': mu_history,
        'k_history': k_history
    }, r'models/inverse_model_intermediate.pth'
    )
    '''

    # Plot parameter evolution
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(mu_history, label="mu_train")
    plt.axhline(y=2 * d, color="r", linestyle="--", label="True mu")
    plt.xlabel("Iteration")
    plt.ylabel("mu_train")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(k_history, label="k_train")
    plt.axhline(y=w ** 2, color="r", linestyle="--", label="True k")
    plt.xlabel("Iteration")
    plt.ylabel("k_train")
    plt.legend()

    plt.tight_layout()
    plt.show()

    print("Training complete.")
    exit()

if goal == objectives[3]:
    torch.manual_seed(123)

    data_saved = torch.load(r'models/inverse_model_almost_done.pth')

    # I am picking data on the entire domain
    x_data_new = x[::2]
    y_data_new = y[::2]

    model = FCN(1, 1, 64, 5, nn.Tanh)
    model.load_state_dict(data_saved['model_state_dict'])

    x_physics = torch.linspace(0, 1, 500).view(-1, 1).requires_grad_(True)

    mu_train = nn.Parameter(torch.tensor(data_saved['mu'].detach(), dtype=torch.float32, requires_grad=True))
    k_train = nn.Parameter(torch.tensor(data_saved['k'].detach(), dtype=torch.float32, requires_grad=True))

    # boundary conditions
    single_point = x[0].unsqueeze(1)
    single_point.requires_grad = True

    # Lists to store the parameter updates
    mu_history = data_saved['mu_history']
    k_history = data_saved['k_history']

    optimizer_full = torch.optim.Adam([
        {'params': model.parameters(), 'lr': 1e-4},
        {'params': [mu_train, k_train], 'lr': 1e-2}  # Higher learning rate for parameters
    ])

    optimizer_full.load_state_dict(data_saved['optimizer_state_dict'])

    for i in range(1000000, 10000000):
        optimizer_full.zero_grad()

        # Solution loss (already trained)
        loss1 = torch.mean((model(x_data_new) - y_data_new) ** 2)
        bc1 = model(single_point)
        bc2 = torch.autograd.grad(bc1, single_point, torch.ones_like(bc1), create_graph=True)[0]
        loss3 = torch.mean((bc1 - 1) ** 2)  # Boundary conditions (if any)
        loss4 = torch.mean(bc2 ** 2)

        # Compute physics loss
        yhp = model(x_physics)  # Predicted solution at physics points
        dx = torch.autograd.grad(yhp, x_physics, torch.ones_like(yhp), create_graph=True, retain_graph=True)[0]
        dx2 = torch.autograd.grad(dx, x_physics, torch.ones_like(dx), create_graph=True, retain_graph=True)[0]
        physics = dx2 + mu_train * dx + k_train * yhp
        loss2 = torch.mean(physics ** 2)  # Physics loss

        # the model tends to go into an overdumped state, therefore I am applying a soft constraint by regularization
        # this approach do not avoid entirely the overdamped state, because it strongly depends on the lambda_reg
        damping_constraint = torch.relu(mu_train - 2 * torch.sqrt(k_train)) ** 2
        lambda_reg = 0.1

        """
        plt.figure()
        plt.plot(x, y, label="Exact solution")
        plt.scatter(x_data_new, y_data_new, color="tab:orange", label="Training data")
        plt.scatter(x_data_new, model(x_data_new).detach().numpy(), color='red', label="Prediction")
        plt.legend()


        plt.show()
        """

        # Total loss
        loss = loss1 + loss2 + loss3 + loss4 + lambda_reg * damping_constraint

        loss.backward()
        optimizer_full.step()

        # Store the values of mu and k
        mu_history.append(mu_train.item())
        k_history.append(k_train.item())

        if i % 100 == 0:
            print(f"Epoch {i}, Loss: {loss.item()}, mu_train: {mu_train.item()}, k_train: {k_train.item()}"
                  f", Solution Loss: {loss1.item()}, PDE Loss: {loss2.item()}, Dirichlet Loss: {loss3.item()}, Neumann Loss: {loss4.item()}"
                  f", Damping Constraint: {damping_constraint.item()}")

        # plot the result as training progresses
        if (i + 1) % 2000 == 0:

            yh = model(x).detach()
            xp = x_physics.detach()

            plot_result(x, y, x_data_new, y_data_new, yh)

            file = "plots/pinn_inverse_%.8i.png" % (i + 1)
            plt.savefig(file, bbox_inches='tight', pad_inches=0.1, dpi=100, facecolor="white")
            #files.append(file)

            if (i + 1) % 50000 == 0:
                plt.show()
            else:
                plt.close("all")

        if (i + 1) % 50000 == 0:
            torch.save({
                'iteration': i,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer_full.state_dict(),
                'loss': loss,
                'data_loss': loss1,
                'physics_loss': loss2,
                'dirichlet_loss': loss3,
                'numann_loss': loss4,
                'mu': mu_train,
                'k': k_train,
                'mu_history': mu_history,
                'k_history': k_history
            }, r'models/inverse_model_intermediate_{}.pth'.format(i+1)
            )
    exit()

    a = 0

if goal == objectives[4]:
    torch.manual_seed(123)

    '''Here I am introducing some optimization to the algorithm, including te LBFGS optimizer, 
       the GELu activation function and the Xavier initialization
    '''

    def closure():
        optimizer_full.zero_grad()

        # Compute solution loss (data loss)
        yh = model(x_data)
        loss1 = torch.mean((yh - y_data) ** 2)

        # Compute physics loss
        yhp = model(x_physics)
        dx = torch.autograd.grad(yhp, x_physics, torch.ones_like(yhp), create_graph=True)[0]
        dx2 = torch.autograd.grad(dx, x_physics, torch.ones_like(dx), create_graph=True)[0]
        physics = dx2 + mu_train * dx + k_train * yhp
        loss2 = torch.mean(physics ** 2) * 1e-3

        # Apply boundary conditions
        single_point = x[0].unsqueeze(1)
        single_point.requires_grad = True
        bc1 = model(single_point)
        bc2 = torch.autograd.grad(bc1, single_point, torch.ones_like(bc1), create_graph=True)[0]

        loss3 = torch.mean((bc1 - 1) ** 2)
        loss4 = torch.mean(bc2 ** 2)

        # Total loss
        loss = loss1 + loss2 + loss3 + loss4
        loss.backward()  # Compute gradients

        return loss


    # I am picking data on the entire domain
    x_data_new = x[::2]
    y_data_new = y[::2]

    model = FCN(1, 1, 64, 5, nn.GELU)

    x_physics = torch.linspace(0, 1, 500).view(-1, 1).requires_grad_(True)

    mu_train = nn.Parameter(torch.tensor(1.0, dtype=torch.float32, requires_grad=True))
    k_train = nn.Parameter(torch.tensor(300.0, dtype=torch.float32, requires_grad=True))

    # boundary conditions
    single_point = x[0].unsqueeze(1)
    single_point.requires_grad = True

    # Optimizer for pretraining (only updates model parameters)
    optimizer_model = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Lists to store the parameter updates
    mu_history = []
    k_history = []
    files = []
    # Pretraining loop
    for epoch in range(6000): #6000
        optimizer_model.zero_grad()

        # Compute losses (solution loss, boundary loss, etc.)
        loss1 = torch.mean((model(x_data_new) - y_data_new) ** 2)  # Data loss
        bc1 = model(single_point)
        bc2 = torch.autograd.grad(bc1, single_point, torch.ones_like(bc1), create_graph=True)[0]
        loss3 = torch.mean((bc1 - 1) ** 2)  # Boundary conditions (if any)
        loss4 = torch.mean(bc2 ** 2)

        loss_pretrain = loss1 + loss3 + loss4  # Pretraining loss

        loss_pretrain.backward()
        optimizer_model.step()

        if epoch % 100 == 0:
            print(f"Pretrain Epoch {epoch}, Loss: {loss_pretrain.item()}")

    print("Pretraining finished. Now optimizing parameters...")

    #optimizer_full = torch.optim.Adam([
    #    {'params': model.parameters(), 'lr': 1e-4},
    #    {'params': [mu_train, k_train], 'lr': 1e-2}  # Higher learning rate for parameters
    #])

    optimizer_full = torch.optim.LBFGS(list(model.parameters()) + [mu_train, k_train], lr=0.5, max_iter=10, history_size=50)

    for i in range(10000000):
        optimizer_full.zero_grad()

        # Solution loss (already trained)
        loss1 = torch.mean((model(x_data_new) - y_data_new) ** 2)
        bc1 = model(single_point)
        bc2 = torch.autograd.grad(bc1, single_point, torch.ones_like(bc1), create_graph=True)[0]
        loss3 = torch.mean((bc1 - 1) ** 2)  # Boundary conditions (if any)
        loss4 = torch.mean(bc2 ** 2)

        # Compute physics loss
        yhp = model(x_physics)  # Predicted solution at physics points
        dx = torch.autograd.grad(yhp, x_physics, torch.ones_like(yhp), create_graph=True, retain_graph=True)[0]
        dx2 = torch.autograd.grad(dx, x_physics, torch.ones_like(dx), create_graph=True, retain_graph=True)[0]
        physics = dx2 + mu_train * dx + k_train * yhp
        loss2 = torch.mean(physics ** 2) * 1e-3# Physics loss

        # the model tends to go into an overdumped state, therefore I am applying a soft constraint by regularization
        # this approach do not avoid entirely the overdamped state, because it strongly depends on the lambda_reg
        damping_constraint = torch.relu(mu_train - 2 * torch.sqrt(k_train)) ** 2
        lambda_reg = 0.1

        # Total loss
        loss = loss1 + loss2 + loss3 + loss4 + lambda_reg * damping_constraint

        loss.backward()
        optimizer_full.step(closure)

        # Store the values of mu and k
        mu_history.append(mu_train.item())
        k_history.append(k_train.item())

        if i % 100 == 0:
            print(f"Epoch {i}, Loss: {loss.item()}, mu_train: {mu_train.item()}, k_train: {k_train.item()}"
                  f", Solution Loss: {loss1.item()}, PDE Loss: {loss2.item()}, Dirichlet Loss: {loss3.item()}, Neumann Loss: {loss4.item()}"
                  f", Damping Constraint: {damping_constraint.item()}")

            for name, param in model.named_parameters():
                if param.grad is not None:
                    print(f"{name}: {torch.norm(param.grad).item()}")

        # plot the result as training progresses
        if (i + 1) % 2000 == 0:

            yh = model(x).detach()
            xp = x_physics.detach()

            plot_result(x, y, x_data_new, y_data_new, yh)

            file = "plots/pinn_inverse_%.8i.png" % (i + 1)
            #plt.savefig(file, bbox_inches='tight', pad_inches=0.1, dpi=100, facecolor="white")
            files.append(file)

            if (i + 1) % 6000 == 0:
                plt.show()
            else:
                plt.close("all")

    #save_gif_PIL("pinn_inverse.gif", files, fps=20, loop=0)

    '''
    torch.save({
        'iteration': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer_full.state_dict(),
        'loss': loss,
        'data_loss': loss1,
        'physics_loss': loss2,
        'dirichlet_loss': loss3,
        'numann_loss': loss4,
        'mu': mu_train,
        'k': k_train,
        'mu_history': mu_history,
        'k_history': k_history
    }, r'models/inverse_model_intermediate.pth'
    )
    '''

    # Plot parameter evolution
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(mu_history, label="mu_train")
    plt.axhline(y=2 * d, color="r", linestyle="--", label="True mu")
    plt.xlabel("Iteration")
    plt.ylabel("mu_train")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(k_history, label="k_train")
    plt.axhline(y=w ** 2, color="r", linestyle="--", label="True k")
    plt.xlabel("Iteration")
    plt.ylabel("k_train")
    plt.legend()

    plt.tight_layout()
    plt.show()

    print("Training complete.")


if goal == objectives[5]:
    torch.manual_seed(123)

    '''
    I did not like muche le lbfgs algorithm so i am going back to the adam optimizer, but i am using pretraining and 
    the gelu function, altogether with weight for the pde 
    '''

    # I am picking data on the entire domain
    x_data_new = x[::2]
    y_data_new = y[::2]

    model = FCN(1, 1, 64, 5, nn.GELU)

    x_physics = torch.linspace(0, 1, 500).view(-1, 1).requires_grad_(True)

    mu_train = nn.Parameter(torch.tensor(1.0, dtype=torch.float32, requires_grad=True))
    k_train = nn.Parameter(torch.tensor(300.0, dtype=torch.float32, requires_grad=True))

    # boundary conditions
    single_point = x[0].unsqueeze(1)
    single_point.requires_grad = True

    # Optimizer for pretraining (only updates model parameters)
    optimizer_model = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Lists to store the parameter updates
    mu_history = []
    k_history = []
    files = []
    # Pretraining loop
    for epoch in range(6000): #6000
        optimizer_model.zero_grad()

        # Compute losses (solution loss, boundary loss, etc.)
        loss1 = torch.mean((model(x_data_new) - y_data_new) ** 2)  # Data loss
        bc1 = model(single_point)
        bc2 = torch.autograd.grad(bc1, single_point, torch.ones_like(bc1), create_graph=True)[0]
        loss3 = torch.mean((bc1 - 1) ** 2)  # Boundary conditions (if any)
        loss4 = torch.mean(bc2 ** 2)

        loss_pretrain = loss1 + loss3 + loss4  # Pretraining loss

        loss_pretrain.backward()
        optimizer_model.step()

        if epoch % 100 == 0:
            print(f"Pretrain Epoch {epoch}, Loss: {loss_pretrain.item()}")

    print("Pretraining finished. Now optimizing parameters...")

    optimizer_full = torch.optim.Adam([
        {'params': model.parameters(), 'lr': 1e-3},
        {'params': [mu_train, k_train], 'lr': 1e-3}  # Higher learning rate for parameters
    ])

    for i in range(10000000):
        optimizer_full.zero_grad()

        # Solution loss (already trained)
        loss1 = torch.mean((model(x_data_new) - y_data_new) ** 2)
        bc1 = model(single_point)
        bc2 = torch.autograd.grad(bc1, single_point, torch.ones_like(bc1), create_graph=True)[0]
        loss3 = torch.mean((bc1 - 1) ** 2)  # Boundary conditions (if any)
        loss4 = torch.mean(bc2 ** 2)

        # Compute physics loss
        yhp = model(x_physics)  # Predicted solution at physics points
        dx = torch.autograd.grad(yhp, x_physics, torch.ones_like(yhp), create_graph=True, retain_graph=True)[0]
        dx2 = torch.autograd.grad(dx, x_physics, torch.ones_like(dx), create_graph=True, retain_graph=True)[0]
        physics = dx2 + mu_train * dx + k_train * yhp

        # Here I am gradually increasing the physics loss weight over training.
        lambda_pde = min(i / 100000, 1.0)  # Linearly increase PDE loss weight
        loss2 = lambda_pde * torch.mean(physics ** 2)

        #loss2 = torch.mean(physics ** 2) * 1e-3 # Physics loss

        # the model tends to go into an overdumped state, therefore I am applying a soft constraint by regularization
        # this approach do not avoid entirely the overdamped state, because it strongly depends on the lambda_reg
        damping_constraint = torch.relu(mu_train - 2 * torch.sqrt(k_train)) ** 2
        lambda_reg = 0.1

        # Total loss
        loss = loss1 + loss2 + loss3 + loss4 + lambda_reg * damping_constraint

        loss.backward()
        optimizer_full.step()

        # Store the values of mu and k
        mu_history.append(mu_train.item())
        k_history.append(k_train.item())

        if i % 100 == 0:
            print(f"Epoch {i}, Loss: {loss.item()}, mu_train: {mu_train.item()}, k_train: {k_train.item()}"
                  f", Solution Loss: {loss1.item()}, PDE Loss: {loss2.item()}, Dirichlet Loss: {loss3.item()}, Neumann Loss: {loss4.item()}"
                  f", Damping Constraint: {damping_constraint.item()}")


        # plot the result as training progresses
        if (i + 1) % 2000 == 0:

            yh = model(x).detach()
            xp = x_physics.detach()

            plot_result(x, y, x_data_new, y_data_new, yh)

            file = "plots/pinn_inverse_%.8i.png" % (i + 1)
            #plt.savefig(file, bbox_inches='tight', pad_inches=0.1, dpi=100, facecolor="white")
            files.append(file)

            if (i + 1) % 6000 == 0:
                plt.show()
            else:
                plt.close("all")

    #save_gif_PIL("pinn_inverse.gif", files, fps=20, loop=0)

    # saving the model here
    '''
    torch.save({
        'iteration': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer_full.state_dict(),
        'loss': loss,
        'data_loss': loss1,
        'physics_loss': loss2,
        'dirichlet_loss': loss3,
        'numann_loss': loss4,
        'mu': mu_train,
        'k': k_train,
        'mu_history': mu_history,
        'k_history': k_history
    }, r'models/inverse_model_intermediate.pth'
    )
    '''

    # Plot parameter evolution
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(mu_history, label="mu_train")
    plt.axhline(y=2 * d, color="r", linestyle="--", label="True mu")
    plt.xlabel("Iteration")
    plt.ylabel("mu_train")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(k_history, label="k_train")
    plt.axhline(y=w ** 2, color="r", linestyle="--", label="True k")
    plt.xlabel("Iteration")
    plt.ylabel("k_train")
    plt.legend()

    plt.tight_layout()
    plt.show()

    print("Training complete.")

a = 0