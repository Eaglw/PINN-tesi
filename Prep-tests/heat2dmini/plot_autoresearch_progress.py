import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_progress(csv_path, output_dir):
    if not os.path.exists(csv_path):
        print(f"CSV not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.sort_values('Timestamp')

    # Distinguish short vs long runs
    # Short: Epochs <= 2000 and L-BFGS <= 500
    # Long: Anything significantly more
    short_runs = df[(df['Epochs'] <= 2000) & (df['Optimizer'].str.contains('500'))].copy()
    long_runs = df[~df.index.isin(short_runs.index)].copy()

    os.makedirs(output_dir, exist_ok=True)

    # Plot Short Runs
    if not short_runs.empty:
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(short_runs)), short_runs['L2_Relative_Error'], marker='o', linestyle='-', color='blue', label='Short Iterations')
        plt.axhline(y=short_runs['L2_Relative_Error'].min(), color='r', linestyle='--', alpha=0.5, label=f"Best: {short_runs['L2_Relative_Error'].min():.4f}")
        plt.title('Autoresearch Progress: Short Test Runs (L2 Error)')
        plt.xlabel('Iteration #')
        plt.ylabel('L2 Relative Error')
        plt.yscale('log')
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'short_runs_progress.png'))
        plt.close()

    # Plot Long/Scale-up Runs
    if not long_runs.empty:
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(long_runs)), long_runs['L2_Relative_Error'], marker='s', linestyle='-', color='green', label='Long/Scale-up Runs')
        plt.axhline(y=long_runs['L2_Relative_Error'].min(), color='r', linestyle='--', alpha=0.5, label=f"Best: {long_runs['L2_Relative_Error'].min():.4f}")
        plt.title('Autoresearch Progress: Long Validation Runs (L2 Error)')
        plt.xlabel('Validation Run #')
        plt.ylabel('L2 Relative Error')
        plt.yscale('log')
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'long_runs_progress.png'))
        plt.close()

    print(f"Plots saved in {output_dir}")

if __name__ == "__main__":
    plot_progress('heat2dmini/mini_results.csv', 'heat2dmini/plots_progress')
