import csv
import os
import shutil
import ast

RESULTS_CSV = 'Heat2D/results.csv'
EXPERIMENTS_DIR = 'Heat2D/experiments'
EPOCH_THRESHOLD = 10000

def parse_architecture(arch_str):
    """
    Parses '[2, 50, 50, 50, 50, 1]' -> '50x4'
    Assumes structure: [Input, Hidden..., Output]
    """
    try:
        arch_list = ast.literal_eval(arch_str)
        # Hidden layers are indices 1:-1
        hidden = arch_list[1:-1]
        if not hidden:
             return "0x0" 
        
        # Check if all hidden have same size
        first_size = hidden[0]
        # For naming convention, we assume standard width even if varying, 
        # but the current project uses fixed width.
        # Naming convention seems to use {Width}x{Depth}.
        return f"{first_size}x{len(hidden)}"
    except:
        return None

def get_candidates():
    candidates = [] 
    
    if not os.path.exists(RESULTS_CSV):
        print(f"Error: {RESULTS_CSV} not found.")
        return []

    with open(RESULTS_CSV, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            try:
                epochs = int(row['Epochs'])
                if epochs < EPOCH_THRESHOLD:
                    # Construct potential folder name parts
                    arch_str = row['Architecture']
                    arch_short = parse_architecture(arch_str) 
                    
                    arch_list = ast.literal_eval(arch_str)
                    output_dim = str(arch_list[-1])
                    
                    activation = row['Activation_Func']
                    
                    # Pattern: L2_{arch}_{out}_E{epochs}_{activation}
                    # Example: L2_50x4_1_E500_GELU
                    base_name = f"L2_{arch_short}_{output_dim}_E{epochs}_{activation}"
                    
                    found_folders = []
                    if os.path.exists(EXPERIMENTS_DIR):
                        for item in os.listdir(EXPERIMENTS_DIR):
                            # Check if item starts with base_name
                            # match exactly or suffix like _step_decay
                            if item.startswith(base_name):
                                found_folders.append(os.path.join(EXPERIMENTS_DIR, item))
                    
                    candidates.append({
                        'index': i, 
                        'row': row, 
                        'folders': found_folders
                    })
            except ValueError:
                continue
                
    return candidates

def main():
    print(f"Scanning {RESULTS_CSV} for experiments with < {EPOCH_THRESHOLD} epochs...")
    candidates = get_candidates()
    
    if not candidates:
        print("No experiments found matching criteria.")
        return

    # Aggregate folders to delete (unique)
    folders_to_delete = set()
    rows_to_delete_indices = []
    
    for c in candidates:
        rows_to_delete_indices.append(c['index'])
        for f in c['folders']:
            folders_to_delete.add(f)

    print(f"\nFound {len(rows_to_delete_indices)} matching entries in CSV.")
    print(f"Identified {len(folders_to_delete)} unique experiment directories.")
    
    print("\n--- Proposed Deletions ---")
    print("Directories:")
    for f in sorted(folders_to_delete):
        print(f"  [DELETE] {f}")
        
    # Validation: warn if a row matched no folder (might be dirty CSV)
    # But we still proceed to clean the CSV.
    
    confirm = input("\nDo you want to proceed with deletion? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Operation cancelled.")
        return

    # 1. Update CSV
    print("Updating CSV...")
    with open(RESULTS_CSV, 'r', newline='') as f:
        all_lines = list(csv.reader(f))
        
    header = all_lines[0]
    data_rows = all_lines[1:]
    
    # Filter
    new_data_rows = [row for i, row in enumerate(data_rows) if i not in rows_to_delete_indices]
    
    with open(RESULTS_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(new_data_rows)
        
    print(f"CSV updated. Removed {len(rows_to_delete_indices)} rows.")
    
    # 2. Delete Folders
    print("Deleting directories...")
    deleted_count = 0
    for f in folders_to_delete:
        if os.path.exists(f):
            try:
                shutil.rmtree(f)
                print(f"  Deleted: {f}")
                deleted_count += 1
            except Exception as e:
                print(f"  Error deleting {f}: {e}")
        else:
            print(f"  Already gone: {f}")
            
    print(f"Cleanup complete. Deleted {deleted_count} directories.")

if __name__ == "__main__":
    main()
