import os
import csv
import torch
import numpy as np

def compute_metrics(model, xy_grid_flat, T_grid_true):
    """
    Computes L2 Relative Error and Max Relative Error Peak.
    
    IMPORTANTE: Questa funzione assume che model(x) restituisca un singolo output
    scalare per punto (shape (N, 1) o (N,)). Per modelli multi-output come
    ViscoelasticCombinedModel (che produce [psi, p, tau_xx, tau_xy, tau_yy]),
    è NECESSARIO wrappare il modello con VelocityInferenceWrapper prima di
    chiamare questa funzione, altrimenti le dimensioni non matchano.
    
    Args:
        model: Trained PyTorch model (single-output o wrapped).
        xy_grid_flat: Tensor of shape (N, 2) containing grid points.
        T_grid_true: Tensor of shape (Nx, Ny) or (N,) containing analytical solution.
    
    Returns:
        l2_rel_error (float): Global L2 relative error norm (ratio).
        max_rel_error_peak (float): Maximum pointwise relative error (percentage).
    """
    model.eval()
    with torch.no_grad():
        # Ensure input has the same dtype as the model weights
        dtype = next(model.parameters()).dtype
        T_pred = model(xy_grid_flat.to(dtype))
        
    # Ensure shapes match (flatten both) and use analytical solution's dtype for metrics
    T_pred_flat = T_pred.view(-1).to(T_grid_true.dtype)
    T_true_flat = T_grid_true.view(-1)
    
    # L2 Relative Error
    # ||u_pred - u_true||_2 / ||u_true||_2
    l2_error = torch.norm(T_pred_flat - T_true_flat, 2)
    l2_ref = torch.norm(T_true_flat, 2)
    
    # Handle division by zero for L2
    if l2_ref > 1e-10:
        l2_rel_error = (l2_error / l2_ref).item()
    else:
        l2_rel_error = 0.0 # Should unlikely happen for Heat Eq solution 
    
    # Max Relative Error Peak
    # Using the same mask logic as in graphic_func.py to avoid division by small numbers
    abs_error = torch.abs(T_pred_flat - T_true_flat)
    mask = torch.abs(T_true_flat) > 0.01
    
    rel_error = torch.zeros_like(T_true_flat)
    
    # Check if mask has any valid values to avoid empty tensor operations
    if mask.sum() > 0:
        # Calculate percentage error
        rel_error[mask] = (abs_error[mask] / torch.abs(T_true_flat[mask])) * 100
        max_rel_error_peak = torch.max(rel_error).item()
    else:
        max_rel_error_peak = 0.0 
        
    return l2_rel_error, max_rel_error_peak

def update_results_csv(file_path, data_dict):
    """
    Appends a row of results to the CSV file.
    
    Args:
        file_path: Path to the CSV file.
        data_dict: Dictionary containing the data to log. 
                   Keys must match the specified columns.
    """
    fieldnames = [
        'Timestamp', 'Max_Relative_Error_Peak', 'Architecture', 'Activation_Func', 'Epochs', 'Run_Type',
        'Optimizer', 'Learning_Rate', 'Loss_Total', 'Loss_Physics', 
        'Loss_Boundary', 'Loss_Data', 'L2_Relative_Error', 'Seed', 'n_points', 'Loss_Weight'
    ]
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    file_exists = os.path.exists(file_path)
    
    try:
        with open(file_path, mode='a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            # Write the row
            writer.writerow(data_dict)
            
    except Exception as e:
        print(f"Error updating CSV log: {e}")

def extract_hyperparams_from_path(path):
    """
    Extracts hyperparameters from a directory path following the naming convention:
    'L<arch>_E<epochs>_<activation>'
    
    Args:
        path: Path string.
        
    Returns:
        tuple: (architecture, epochs, activation)
    """
    parts = os.path.normpath(path).split(os.sep)
    target = None
    # Look for the segment that follows our convention
    for p in reversed(parts):
        if p.startswith('L') and '_E' in p:
            target = p
            break
            
    if not target:
        return "N/A", "N/A", "N/A"
        
    try:
        # Example: L2_50x4_1_E20000_GELU
        # Find index of _E
        idx_e = target.find('_E')
        arch = target[1:idx_e] # 2_50x4_1
        
        rest = target[idx_e+2:] # 20000_GELU
        if '_' in rest:
            split_rest = rest.split('_')
            epochs = split_rest[0]
            activation = split_rest[1]
        else:
            epochs = rest
            activation = "N/A"
            
        return arch, epochs, activation
    except Exception:
        return "Error", "Error", "Error"
