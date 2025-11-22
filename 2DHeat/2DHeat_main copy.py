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




def soluzione_analitica(x, y, Lx, Ly, Nx=50):
    """
    Soluzione analitica della temperatura in una lastra rettangolare 2D
    con Dirichlet: T=0 su y=0, y=Ly, x=0; T=1 su x=Lx.
    x, y possono essere array di eguali dimensioni.
    """
    T = np.zeros_like(x, dtype=float)
    for n in range(1, Nx+1):
        lambda_n = n * np.pi / Ly
        An = 2 * (1 - (-1)**n) / (n * np.pi)
        T += An * np.sinh(lambda_n * x) / np.sinh(lambda_n * Lx) * np.sin(lambda_n * y)
    return T

# Parametri dominio
Lx, Ly = 1.0, 1.0
Nx_fourier = 50  # termini serie

# Griglia dominio per visualizzazione
Nx_dom, Ny_dom = 100, 100
x_grid = np.linspace(0, Lx, Nx_dom)
y_grid = np.linspace(0, Ly, Ny_dom)
X, Y = np.meshgrid(x_grid, y_grid)

T_grid = soluzione_analitica(X, Y, Lx, Ly, Nx=Nx_fourier)

# Estrazione dati randomici ma uniformi
num_data = 200  # cambia a piacere
np.random.seed(0)

x_data = np.random.uniform(0, Lx, num_data)
y_data = np.random.uniform(0, Ly, num_data)
T_data = soluzione_analitica(x_data, y_data, Lx, Ly, Nx=Nx_fourier)

# Plot
plt.figure(figsize=(8,6))
cp = plt.contourf(X, Y, T_grid, 50, cmap='inferno')
plt.colorbar(cp)
plt.scatter(x_data, y_data, c='cyan', s=21, edgecolor='k', label='Dati estratti')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.title('Soluzione analitica e punti dati estratti')
plt.legend()
plt.show()
