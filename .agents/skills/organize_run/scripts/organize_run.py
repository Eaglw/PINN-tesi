import os
import sys
import re

def parse_log(log_path):
    if not os.path.exists(log_path):
        return {"status": "EMPTY", "final_l2_u": None, "final_l2_p": None, "epochs": None}
    
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
        
    status = "COMPLETED"
    if 'KeyboardInterrupt' in content:
        status = "CRASHED (KeyboardInterrupt)"
    elif 'NameError' in content:
        status = "CRASHED (NameError)"
    elif 'ValueError' in content:
        status = "CRASHED (ValueError)"
    elif 'CUDA out of memory' in content:
        status = "CRASHED (OOM)"
    elif 'Traceback' in content:
        status = "CRASHED (Exception)"
    elif '[OK] Esecuzione terminata' not in content and 'L2 Errors' not in content:
        status = "INCOMPLETE"
        
    # Extract L2 errors
    l2_matches = re.findall(r'L2 Errors -> u: ([\d\.e+-]+) \| v: ([\d\.e+-]+) \| p: ([\d\.e+-]+)', content)
    l2_u = float(l2_matches[-1][0]) if l2_matches else None
    l2_p = float(l2_matches[-1][2]) if l2_matches else None
    
    # Extract epoch count
    epochs_matches = re.findall(r'Epoch (\d+)', content)
    max_epoch = max([int(e) for e in epochs_matches]) if epochs_matches else None
    
    return {
        "status": status,
        "content": content,
        "l2_u": l2_u,
        "l2_p": l2_p,
        "max_epoch": max_epoch
    }

def process_output_directory(output_dir):
    if not os.path.exists(output_dir):
        print(f"Directory non trovata: {output_dir}")
        return

    folders = [f for f in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, f))]
    print(f"Scansione di {len(folders)} cartelle in {output_dir}...")
    
    for f in sorted(folders):
        # Ignora cartelle già nel formato standard [PREFISSO]
        if f.startswith("["):
            continue
            
        full_path = os.path.join(output_dir, f)
        log_path = os.path.join(full_path, "train_log.txt")
        info = parse_log(log_path)
        
        png_count = len([x for x in os.listdir(full_path) if x.endswith('.png')])
        has_ckpt = os.path.exists(os.path.join(full_path, "checkpoint.pth"))
        
        print(f"\nAnalisi per: {f}")
        print(f"  Status: {info['status']} | L2 u: {info['l2_u']} | L2 p: {info['l2_p']} | Plots: {png_count} | Ckpt: {has_ckpt}")

if __name__ == "__main__":
    target_dir = sys.argv[1] if len(sys.argv) > 1 else r"c:\Users\eaglw\Documents\PINN tesi\final_roll\output_4rollmill"
    process_output_directory(target_dir)
