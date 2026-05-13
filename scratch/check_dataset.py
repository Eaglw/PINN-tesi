import torch
import os

dataset_path = r'c:\Users\eaglw\Documents\PINN tesi\Viscoelastic\dataset\oldroydb_clean.pt'
if not os.path.exists(dataset_path):
    print("Dataset not found")
else:
    data = torch.load(dataset_path, weights_only=False)
    print("--- Dataset Check ---")
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            nan_count = torch.isnan(v).sum().item()
            print(f"{k}: shape={v.shape}, NaNs={nan_count}")
            if nan_count > 0:
                print(f"  ERROR: {k} contains NaNs!")
    
    params = data['params']
    print("\n--- Params Check ---")
    for k, v in params.items():
        print(f"{k}: {v}")
