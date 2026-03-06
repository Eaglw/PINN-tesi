import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def visualize_results(csv_path='Heat2D-fp64-32/precision_benchmark_results.csv', output_dir='Heat2D-fp64-32/benchmark_plots'):
    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    
    # 1. Error vs Speedup Scatter Plot
    # Usiamo 'nn_opt' (Net & Optimizer) come colore e 'physics' come dimensione del punto
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='Speedup', y='MAE_Gold', hue='nn_opt', style='physics', s=100, alpha=0.8)
    plt.yscale('log')
    plt.title('Error (Relative to Gold) vs Speedup')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.savefig(os.path.join(output_dir, 'error_vs_speedup.png'))
    
    # 2. Precision Sensitivity Analysis
    # Identifichiamo quali componenti contribuiscono di più all'errore
    parts = ['nn_opt', 'data', 'physics', 'bc']
    
    sensitivity = []
    for part in parts:
        avg_fp32 = df[df[part] == 'FP32']['MAE_Gold'].mean()
        avg_fp64 = df[df[part] == 'FP64']['MAE_Gold'].mean()
        sensitivity.append({
            'Component': part,
            'Avg MAE (FP32)': avg_fp32,
            'Avg MAE (FP64)': avg_fp64,
            'Sensitivity (Ratio)': avg_fp32 / avg_fp64 if avg_fp64 > 0 else 1.0
        })
    
    sens_df = pd.DataFrame(sensitivity)
    plt.figure(figsize=(10, 5))
    sns.barplot(data=sens_df, x='Component', y='Sensitivity (Ratio)', palette='viridis')
    plt.yscale('log')
    plt.title('Precision Sensitivity by Component (Ratio of Avg MAE FP32/FP64)')
    plt.ylabel('Sensitivity Ratio (Higher = More Sensitive to Precision)')
    plt.grid(axis='y', ls="--", alpha=0.5)
    plt.savefig(os.path.join(output_dir, 'component_sensitivity.png'))
    
    # 3. Pareto Optimal Candidates
    # Configurazioni con Errore < 2% (rispetto all'errore del Gold) e Massimo Speedup
    # Definiamo una soglia di tolleranza ragionevole
    threshold = df[df['mask'] == 15]['MAE_Analytic'].iloc[0] * 1.05 # 5% tolleranza sopra l'analitico
    
    candidates = df[df['MAE_Analytic'] <= threshold].sort_values(by='Speedup', ascending=False)
    
    print("\n--- Pareto Optimal Candidates (Error within 5% of Gold) ---")
    if not candidates.empty:
        print(candidates[['mask', 'Speedup', 'MAE_Gold', 'nn_opt', 'physics', 'data', 'bc']].head(5))
    else:
        # Se nessuno è sotto la soglia stretta, mostriamo i migliori 5 per speedup
        print("No candidates strictly within 5% tolerance. Showing best speedup results:")
        print(df.sort_values(by='MAE_Gold').head(5)[['mask', 'Speedup', 'MAE_Gold', 'nn_opt', 'physics']])
    
    candidates.to_csv(os.path.join(output_dir, 'pareto_candidates.csv'), index=False)
    print(f"\nPlots and analysis saved to {output_dir}")

if __name__ == "__main__":
    visualize_results()
