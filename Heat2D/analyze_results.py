"""
Heat2D Results Analysis Script
==============================
This script processes 'results.csv' from the Heat2D experiments and generates
comprehensive visualizations and summary statistics to evaluate model performance.

Usage:
    python Heat2D/analyze_results.py
"""

import pandas as pd
import ast
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns

# Set global style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

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

def plot_method_comparison(df, output_dir):
    """
    Generates bar charts comparing Max_Relative_Error_Peak and L2_Relative_Error
    grouped by Run_Type.
    
    Args:
        df (pd.DataFrame): The results dataframe.
        output_dir (str): Directory to save plots.
    """
    if df is None or output_dir is None:
        return

    metrics = ['Max_Relative_Error_Peak', 'L2_Relative_Error']
    
    for metric in metrics:
        if metric not in df.columns:
            continue
            
        plt.figure(figsize=(12, 8))
        
        # Calculate mean error for each Run_Type to sort the bars
        order = df.groupby('Run_Type')[metric].mean().sort_values().index
        
        sns.barplot(x='Run_Type', y=metric, hue='Run_Type', data=df, order=order, palette='viridis', legend=False)
        
        plt.title(f'Comparison of {metric} by Method', fontsize=16)
        plt.xlabel('Method (Run_Type)', fontsize=14)
        plt.ylabel(metric, fontsize=14)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        filename = f"comparison_{metric}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300)
        plt.close()
        print(f"Saved plot: {filepath}")

def plot_error_correlation(df, output_dir):
    """
    Generates scatter plots to show correlations between metrics.
    1. Loss_Total vs Max_Relative_Error_Peak
    2. Epochs vs Max_Relative_Error_Peak (if Epochs vary)
    
    Args:
        df (pd.DataFrame): The results dataframe.
        output_dir (str): Directory to save plots.
    """
    if df is None or output_dir is None:
        return

    # 1. Loss_Total vs Max_Relative_Error_Peak
    if 'Loss_Total' in df.columns and 'Max_Relative_Error_Peak' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=df, 
            x='Loss_Total', 
            y='Max_Relative_Error_Peak', 
            hue='Run_Type', 
            style='Run_Type', 
            s=100
        )
        plt.xscale('log') # Loss is often logarithmic
        plt.yscale('log') # Error can also span orders of magnitude
        plt.title('Loss Total vs Max Relative Error Peak (Log-Log)', fontsize=16)
        plt.xlabel('Loss Total', fontsize=14)
        plt.ylabel('Max Relative Error Peak', fontsize=14)
        plt.tight_layout()
        
        filepath = os.path.join(output_dir, "correlation_loss_vs_error.png")
        plt.savefig(filepath, dpi=300)
        plt.close()
        print(f"Saved plot: {filepath}")

    # 2. Epochs vs Accuracy (Max_Relative_Error_Peak)
    if 'Epochs' in df.columns and 'Max_Relative_Error_Peak' in df.columns:
        # Check if Epochs vary
        if df['Epochs'].nunique() > 1:
            plt.figure(figsize=(10, 6))
            sns.lineplot(
                data=df,
                x='Epochs',
                y='Max_Relative_Error_Peak',
                hue='Run_Type',
                marker='o',
                err_style='bars' # Show error bars if multiple seeds
            )
            plt.title('Effect of Training Epochs on Accuracy', fontsize=16)
            plt.xlabel('Epochs', fontsize=14)
            plt.ylabel('Max Relative Error Peak', fontsize=14)
            plt.tight_layout()
            
            filepath = os.path.join(output_dir, "correlation_epochs_vs_error.png")
            plt.savefig(filepath, dpi=300)
            plt.close()
            print(f"Saved plot: {filepath}")

def plot_stability_distribution(df, output_dir):
    """
    Generates box plots to visualize the stability of each method (Run_Type)
    across multiple seeds.
    
    Args:
        df (pd.DataFrame): The results dataframe.
        output_dir (str): Directory to save plots.
    """
    if df is None or output_dir is None:
        return

    # Check if we have multiple seeds to make this meaningful
    # Even if not, a box plot is still useful (just looks like a line/point)
    
    metrics = ['Max_Relative_Error_Peak', 'L2_Relative_Error']
    
    for metric in metrics:
        if metric not in df.columns:
            continue
            
        plt.figure(figsize=(12, 8))
        
        # Calculate mean error for each Run_Type to sort the bars
        order = df.groupby('Run_Type')[metric].mean().sort_values().index
        
        sns.boxplot(x='Run_Type', y=metric, data=df, order=order, palette='viridis', hue='Run_Type', legend=False)
        sns.swarmplot(x='Run_Type', y=metric, data=df, order=order, color=".25", size=5) # Show individual points
        
        plt.title(f'Stability Distribution of {metric} by Method', fontsize=16)
        plt.xlabel('Method (Run_Type)', fontsize=14)
        plt.ylabel(metric, fontsize=14)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        filename = f"stability_{metric}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300)
        plt.close()
        print(f"Saved plot: {filepath}")

def plot_hyperparam_heatmap(df, output_dir):
    """
    Generates heatmaps to show performance across different hyperparameters.
    Architecture vs Activation_Func.
    
    Args:
        df (pd.DataFrame): The results dataframe.
        output_dir (str): Directory to save plots.
    """
    if df is None or output_dir is None:
        return

    # Check for required columns
    if not all(col in df.columns for col in ['Architecture', 'Activation_Func', 'Max_Relative_Error_Peak']):
        return

    # Prepare a copy for heatmap
    heatmap_df = df.copy()
    
    # Convert Architecture list to string for labeling
    heatmap_df['Arch_Str'] = heatmap_df['Architecture'].apply(lambda x: str(x))
    
    # Create pivot table for mean error
    pivot_table = heatmap_df.pivot_table(
        values='Max_Relative_Error_Peak', 
        index='Arch_Str', 
        columns='Activation_Func', 
        aggfunc='mean'
    )
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap='YlGnBu', cbar_kws={'label': 'Mean Max Relative Error Peak'})
    
    plt.title('Performance Heatmap: Architecture vs Activation Function', fontsize=16)
    plt.xlabel('Activation Function', fontsize=14)
    plt.ylabel('Architecture', fontsize=14)
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, "heatmap_arch_vs_activation.png")
    plt.savefig(filepath, dpi=300)
    plt.close()
    print(f"Saved plot: {filepath}")

def print_summary_statistics(df):
    """
    Prints key insights and summary statistics to the console.
    
    Args:
        df (pd.DataFrame): The results dataframe.
    """
    if df is None:
        return

    print("\n" + "="*50)
    print("SUMMARY STATISTICS & INSIGHTS")
    print("="*50)

    # Best performing method overall (by Max_Relative_Error_Peak)
    if 'Max_Relative_Error_Peak' in df.columns:
        best_run = df.loc[df['Max_Relative_Error_Peak'].idxmin()]
        print(f"Overall Best Run (Lowest Max Error):")
        print(f"  - Max Relative Error: {best_run['Max_Relative_Error_Peak']:.4f}")
        print(f"  - Run Type:           {best_run['Run_Type']}")
        print(f"  - Architecture:       {best_run['Architecture']}")
        print(f"  - Activation:         {best_run['Activation_Func']}")
        print(f"  - Epochs:             {best_run['Epochs']}")
        print(f"  - Timestamp:          {best_run['Timestamp']}")
        print("-" * 30)

    # Average performance by Run_Type
    if 'Run_Type' in df.columns and 'Max_Relative_Error_Peak' in df.columns:
        avg_by_method = df.groupby('Run_Type')['Max_Relative_Error_Peak'].agg(['mean', 'std', 'min', 'count'])
        avg_by_method = avg_by_method.sort_values(by='mean')
        
        print("Performance Ranking by Method (Run_Type):")
        print(avg_by_method.to_string(formatters={'mean': '{:,.4f}'.format, 'std': '{:,.4f}'.format, 'min': '{:,.4f}'.format}))
        print("-" * 30)

    # Activation Function Comparison
    if 'Activation_Func' in df.columns and 'Max_Relative_Error_Peak' in df.columns:
        avg_by_act = df.groupby('Activation_Func')['Max_Relative_Error_Peak'].mean().sort_values()
        print("Mean Max Error by Activation Function:")
        for act, val in avg_by_act.items():
            print(f"  - {act}: {val:.4f}")
        print("-" * 30)

    print("="*50 + "\n")

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
            
            if output_dir:
                print("Generating method comparison plots...")
                plot_method_comparison(df, output_dir)
                
                print("Generating correlation plots...")
                plot_error_correlation(df, output_dir)
                
                print("Generating stability plots...")
                plot_stability_distribution(df, output_dir)
                
                print("Generating hyperparameter heatmaps...")
                plot_hyperparam_heatmap(df, output_dir)
                
                print("Printing summary statistics...")
                print_summary_statistics(df)
            
            print("-" * 30)
            print("Analysis complete.")
    else:
        print("Error: results.csv not found in common locations.")
