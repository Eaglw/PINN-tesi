import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Per importare le funzioni dalla cartella func, si aggiunge il percorso della cartella genitore
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Le funzioni in func/graphic_func.py sono per problemi 1D. 
# Si definisce qui una funzione di plotting specifica per il caso 2D.
# In futuro, questa potrebbe essere generalizzata e spostata in graphic_func.py.

def plot_2d_heat_solution(x_grid, y_grid, T_solution, data_points=None, title="Soluzione di Temperatura 2D"):
    """
    Crea un contour plot per la soluzione della temperatura 2D e, opzionalmente,
    sovrappone i punti di dati sperimentali.

    Args:
        x_grid (torch.Tensor): Griglia di coordinate x (2D).
        y_grid (torch.Tensor): Griglia di coordinate y (2D).
        T_solution (torch.Tensor): Matrice della temperatura sulla griglia (2D).
        data_points (tuple, optional): Una tupla (x_data, y_data, T_data) con i punti
                                       sperimentali da plottare. Defaults to None.
        title (str, optional): Titolo del grafico. Defaults to "Soluzione di Temperatura 2D".
    """
    plt.figure(figsize=(10, 6))
    
    # Contour plot della soluzione completa
    contour = plt.contourf(x_grid, y_grid, T_solution, levels=50, cmap='hot')
    plt.colorbar(contour, label='Temperatura (°C)')
    
    # Plot dei punti di dati sperimentali, se forniti
    if data_points is not None:
        x_data, y_data, _ = data_points
        plt.scatter(x_data, y_data, c='blue', edgecolors='k', s=60, label='Dati Sperimentali')
        plt.legend()
        
    plt.title(title)
    plt.xlabel("Posizione x (m)")
    plt.ylabel("Posizione y (m)")
    plt.axis('equal')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Salvataggio del grafico
    output_dir = "plots/2DHeat"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "analytical_solution_with_data.png"), bbox_inches='tight')
    
    plt.show()

def get_analytical_solution(x, y, L, T_left, T_right):
    """
    Calcola la soluzione analitica per l'equazione di Laplace 2D con BC semplici:
    - T(0, y) = T_left
    - T(L, y) = T_right
    - ∂T/∂y(x, 0) = 0 e ∂T/∂y(x, W) = 0
    
    Per queste specifiche BC, la soluzione è una semplice funzione lineare di x.
    """
    # Spostiamo l'origine a (0,0) per semplicità
    x_shifted = x - x.min()
    # Calcolo lineare della temperatura
    return T_left + (T_right - T_left) * (x_shifted / L)

if __name__ == "__main__":
    # --- 1. Definizione dei parametri del problema ---
    L = 2.0  # Lunghezza della lastra in x [m]
    W = 1.0  # Larghezza della lastra in y [m]
    T_left = 100.0  # Temperatura sulla faccia sinistra [°C]
    T_right = 20.0  # Temperatura sulla faccia destra [°C]

    # --- 2. Generazione della griglia per la soluzione completa ---
    nx, ny = 100, 50  # Numero di punti della griglia
    x = torch.linspace(0, L, nx)
    y = torch.linspace(0, W, ny)
    x_grid, y_grid = torch.meshgrid(x, y, indexing='ij')

    # --- 3. Calcolo della soluzione analitica completa ---
    T_analytical = get_analytical_solution(x_grid, y_grid, L, T_left, T_right)
    print("Dimensioni della griglia di soluzione:", T_analytical.shape)

    # --- 4. Estrazione dei "Dati Sperimentali" ---
    # Selezioniamo un sottoinsieme di punti dalla griglia
    num_data_points_x = 10
    num_data_points_y = 5
    
    x_data_idx = torch.linspace(0, nx - 1, num_data_points_x).long()
    y_data_idx = torch.linspace(0, ny - 1, num_data_points_y).long()
    
    data_grid_x, data_grid_y = torch.meshgrid(x_data_idx, y_data_idx, indexing='ij')

    x_data = x_grid[data_grid_x, data_grid_y].flatten()
    y_data = y_grid[data_grid_x, data_grid_y].flatten()
    T_data = T_analytical[data_grid_x, data_grid_y].flatten()

    print(f"Estratti {T_data.numel()} punti di dati sperimentali.")
    # Opzionale: aggiungere un po' di rumore gaussiano per simulare dati reali
    # noise = 0.02 * torch.randn_like(T_data) * (T_left - T_right)
    # T_data += noise

    # --- 5. Plottaggio dei risultati ---
    plot_2d_heat_solution(
        x_grid.T,  # Transpose per la visualizzazione con contourf
        y_grid.T,
        T_analytical.T,
        data_points=(x_data, y_data, T_data),
        title="Soluzione Analitica e Dati Sperimentali Estratti"
    )

    print("Script completato. Grafico salvato in 'plots/2DHeat/'.")
