import torch

def get_gpu_info():
    """
    Rileva le caratteristiche della GPU corrente e consiglia una configurazione di precisione.
    """
    if not torch.cuda.is_available():
        return {
            "name": "CPU",
            "supports_bf16": False,
            "vram_gb": 0,
            "recommended_preset": "FP32_PURE (CPU is slow for PINNs)"
        }
    
    device_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    # Check for bfloat16 support (Ampere architecture or newer: compute capability >= 8.0)
    major, minor = torch.cuda.get_device_capability(0)
    supports_bf16 = major >= 8
    
    # Recommendation logic
    if supports_bf16:
        preset = "HYBRID_BF16 (Weights: BF16, Physics: FP64)"
    elif major >= 6: # Pascal (1050Ti is 6.1)
        preset = "HYBRID_FP32 (Weights: FP32, Physics: FP64)"
    else:
        preset = "FP32_PURE"
        
    return {
        "name": device_name,
        "supports_bf16": supports_bf16,
        "vram_gb": round(vram_gb, 2),
        "compute_capability": f"{major}.{minor}",
        "recommended_preset": preset
    }

if __name__ == "__main__":
    info = get_gpu_info()
    print("\n--- Hardware Detection ---")
    for k, v in info.items():
        print(f"{k.replace('_', ' ').title()}: {v}")
    print("--------------------------\n")
