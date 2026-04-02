import subprocess
import pandas as pd
import os
import sys

def run_verify():
    # Use the venv python if available
    python_exe = sys.executable
    
    cmd = [
        python_exe, "heat2dmini/Heat2D_adaptive_mini.py",
        "--epochs", "1000",
        "--arch", "120,100,80,60,40,20",
        "--bc_weight", "50",
        "--act", "GELU",
        "--lbfgs_iter", "300"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            # Output stderr to help debugging but only the number counts
            # sys.stderr.write(result.stderr)
            pass

        csv_path = "heat2dmini/mini_results.csv"
        if not os.path.exists(csv_path):
            print("999.9")
            return

        df = pd.read_csv(csv_path)
        if df.empty:
            print("999.9")
            return
            
        last_l2 = df.iloc[-1]['L2_Relative_Error']
        print(f"{float(last_l2):.8f}")
        
    except Exception as e:
        print("999.9")

if __name__ == "__main__":
    run_verify()
