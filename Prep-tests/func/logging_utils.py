import os
import csv
import torch
import numpy as np



def update_results_csv(file_path, data_dict):
    """
    Appends a row of results to the CSV file.
    
    Args:
        file_path: Path to the CSV file.
        data_dict: Dictionary containing the data to log. 
                   Keys must match the specified columns.
    """
    fieldnames = [
        'Timestamp', 'Dataset', 'Architecture', 'Activation_Func', 'Epochs', 'Run_Type',
        'Optimizer', 'Learning_Rate', 'Loss_Total', 'Loss_Physics', 
        'Loss_Boundary', 'Loss_Data', 'L2_Relative_Error', 'Max_Relative_Error_Peak',
        'L2_u', 'Max_u', 'L2_p', 'Max_p',
        'L2_tau_xx', 'Max_tau_xx', 'L2_tau_xy', 'Max_tau_xy', 'L2_tau_yy', 'Max_tau_yy',
        'Seed', 'n_points', 'Loss_Weight'
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
