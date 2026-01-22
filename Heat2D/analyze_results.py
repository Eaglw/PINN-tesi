import pandas as pd
import ast
import os
import sys

def load_data(file_path):
    """
    Loads the results CSV file and performs basic data cleaning.
    
    Args:
        file_path (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame: Cleaned dataframe or None if loading fails.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return None

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None

    # Clean Architecture column: Convert string representation of list to actual list
    if 'Architecture' in df.columns:
        try:
            # Only apply literal_eval if it's a string looking like a list
            df['Architecture'] = df['Architecture'].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) and x.strip().startswith('[') else x
            )
        except Exception as e:
            print(f"Warning: Could not parse 'Architecture' column: {e}")

    # Check for missing values
    if df.isnull().values.any():
        print("Warning: Missing values found in the dataset.")
        print(df.isnull().sum())

    return df

def setup_output_dir(base_dir="Heat2D"):
    """
    Creates the output directory for analysis plots.
    
    Args:
        base_dir (str): Base directory where analysis_plots will be created.
        
    Returns:
        str: Path to the output directory.
    """
    output_dir = os.path.join(base_dir, "analysis_plots")
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        except Exception as e:
            print(f"Error creating output directory: {e}")
            return None
    else:
        print(f"Output directory already exists: {output_dir}")
    return output_dir

if __name__ == "__main__":
    # Determine the project root to find the file relative to it
    # This allows running from root or Heat2D folder
    
    possible_paths = [
        "Heat2D/results.csv",
        "results.csv",
        "../Heat2D/results.csv"
    ]
    
    file_path = None
    base_dir = "Heat2D" # Default base dir
    
    for path in possible_paths:
        if os.path.exists(path):
            file_path = path
            # Update base_dir based on where we found the file
            if "Heat2D" not in path and "results.csv" == path:
                 base_dir = "." # Running inside Heat2D
            elif "../" in path:
                 base_dir = "../Heat2D"
            break
            
    if file_path:
        print(f"Loading data from: {file_path}")
        df = load_data(file_path)
        if df is not None:
            print("Data loaded successfully.")
            
            output_dir = setup_output_dir(base_dir)
            
            print("-" * 30)
            print("First 5 rows:")
            print(df.head())
            print("-" * 30)
            print("Data Info:")
            print(df.info())
    else:
        print("Error: results.csv not found in common locations.")
