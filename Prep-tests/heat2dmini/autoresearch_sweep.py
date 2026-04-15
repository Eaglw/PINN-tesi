import os
import subprocess
import itertools
import json

# Define the search space
archs = ["100,100,100,100", "80,80,80,80,80,80", "150,150,150"]
acts = ["Tanh", "SiLU", "GELU"]
update_freqs = [100, 500]
run_types = ["PINN_PurePhys", "PINN_DataPhys"]

combinations = list(itertools.product(archs, acts, update_freqs, run_types))

print(f"Starting Autoresearch Sweep: {len(combinations)} combinations.")

for i, (arch, act, freq, run_type) in enumerate(combinations):
    print(f"\n--- Iteration {i+1}/{len(combinations)} ---")
    print(f"Arch: {arch}, Act: {act}, Freq: {freq}, Type: {run_type}")
    
    cmd = [
        "python", "heat2dmini/Heat2D_weighted_mini.py",
        "--arch", arch,
        "--act", act,
        "--update_weights_every", str(freq),
        "--run_type", run_type,
        "--epochs", "2000",
        "--lbfgs_iter", "500"
    ]
    
    try:
        # Run and capture output
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error in iteration {i+1}:")
            print(result.stderr)
        else:
            # Look for L2 Error in output
            for line in result.stdout.split('\n'):
                if "L2 Relative Error:" in line:
                    print(line)
    except Exception as e:
        print(f"Exception in iteration {i+1}: {e}")

print("\nSweep Complete. Check Heat2D/mini_results.csv for details.")
