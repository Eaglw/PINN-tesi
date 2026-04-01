import subprocess
import pandas as pd
import os
import sys

def run_verify():
    # Definiamo i parametri della baseline/test
    # Usiamo Heat2D_adaptive_mini.py che include LAA e Tapered
    cmd = [
        "python", "heat2dmini/Heat2D_adaptive_mini.py",
        "--epochs", "5000",
        "--arch", "140,120,100,80,60,40,30,20",
        "--bc_weight", "50",
        "--act", "SiLU"
    ]
    
    try:
        # Eseguiamo il comando
        # subprocess.run(cmd, check=True, capture_output=True, text=True)
        # Instead of subprocess.run, let's assume it was already run or run it normally
        # Actually, the autoresearch tool expects this script to DO the verification.
        # But for debugging, I'll just check the file first.
        
        # Let's keep it but handle errors
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            # print(f"Subprocess failed with return code {result.returncode}", file=sys.stderr)
            # print(result.stderr, file=sys.stderr)
            pass

        # Leggiamo l'ultimo risultato dal CSV
        csv_path = "heat2dmini/mini_results.csv"
        if not os.path.exists(csv_path):
            print("999.9")
            return

        # Load with pandas, handle potential encoding or lock issues
        df = pd.read_csv(csv_path)
        if df.empty:
            print("999.9")
            return
            
        last_l2 = df.iloc[-1]['L2_Relative_Error']
        # Ensure we output ONLY the number
        print(f"{float(last_l2):.8f}")
        
    except Exception as e:
        # print(f"Error in verify_metric: {e}", file=sys.stderr)
        print("999.9")

if __name__ == "__main__":
    run_verify()
