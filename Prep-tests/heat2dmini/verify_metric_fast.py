import subprocess
import csv
import os
import sys

def run_verify():
    # Run the main script
    # We use the same python executable
    cmd = [sys.executable, "heat2dmini/Heat2D_adaptive_mini.py"]
    
    # Run and wait
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running script: {result.stderr}")
        sys.exit(1)
        
    # Read the last line from mini_results.csv
    results_csv = "heat2dmini/mini_results.csv"
    if not os.path.exists(results_csv):
        print("Results CSV not found")
        sys.exit(1)
        
    with open(results_csv, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if not rows:
            print("No results found in CSV")
            sys.exit(1)
        
        last_row = rows[-1]
        print(last_row['L2_Relative_Error'])

if __name__ == "__main__":
    run_verify()
