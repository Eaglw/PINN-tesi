import inspect
import sys
import os

# Add project root to sys.path
sys.path.append(os.getcwd())

from Heat2D.src.Heat2D_NN import train_modelNN
from Heat2D.src.Heat2D_NN_griglia import train_modelNN_griglia
from Heat2D.src.Heat2D_PINN import train_modelPINN

def check_signature(func, func_name):
    sig = inspect.signature(func)
    if 'lr_strategy' in sig.parameters:
        print(f"[PASS] {func_name} has 'lr_strategy' parameter.")
        return True
    else:
        print(f"[FAIL] {func_name} MISSING 'lr_strategy' parameter.")
        return False

def main():
    print("Verifying function signatures...")
    all_passed = True
    all_passed &= check_signature(train_modelNN, "train_modelNN")
    all_passed &= check_signature(train_modelNN_griglia, "train_modelNN_griglia")
    all_passed &= check_signature(train_modelPINN, "train_modelPINN")
    
    if all_passed:
        print("\nAll checks passed.")
        exit(0)
    else:
        print("\nSome checks failed.")
        exit(1)

if __name__ == "__main__":
    main()

