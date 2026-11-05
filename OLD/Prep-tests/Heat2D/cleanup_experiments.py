import csv
import os
import shutil
import ast
import re

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
    
    if os.path.exists(RESULTS_CSV):
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
    else:
        print(f"Warning: {RESULTS_CSV} not found. Skipping CSV scan.")
                
    return candidates

def scan_orphan_folders(existing_candidates_folders):
    """
    Scans the experiment directory for folders matching the pattern but not linked to the CSV entries found so far.
    Checks if the epoch count in the folder name is < EPOCH_THRESHOLD.
    """
    orphans = []
    if not os.path.exists(EXPERIMENTS_DIR):
        return orphans

    # Pattern to extract epochs: ..._E{epochs}_...
    # Looks for "_E" followed by digits
    epoch_pattern = re.compile(r'_E(\d+)_')

    for item in os.listdir(EXPERIMENTS_DIR):
        full_path = os.path.join(EXPERIMENTS_DIR, item)
        if not os.path.isdir(full_path):
            continue
            
        # Check if already marked for deletion via CSV
        if full_path in existing_candidates_folders:
            continue

        match = epoch_pattern.search(item)
        if match:
            try:
                epochs = int(match.group(1))
                if epochs < EPOCH_THRESHOLD:
                    orphans.append(full_path)
            except ValueError:
                continue
    
    return orphans

def main():
    print(f"Scanning {RESULTS_CSV} for experiments with < {EPOCH_THRESHOLD} epochs...")
    candidates = get_candidates()
    
    # Aggregate folders from CSV candidates
    folders_from_csv = set()
    rows_to_delete_indices = []
    
    for c in candidates:
        rows_to_delete_indices.append(c['index'])
        for f in c['folders']:
            folders_from_csv.add(f)

    # Scan for orphans
    print("Scanning filesystem for orphan directories...")
    orphans = scan_orphan_folders(folders_from_csv)
    
    all_folders_to_delete = folders_from_csv.union(orphans)

    if not rows_to_delete_indices and not all_folders_to_delete:
        print("No experiments found matching criteria.")
        return

    print(f"\nFound {len(rows_to_delete_indices)} matching entries in CSV.")
    print(f"Identified {len(all_folders_to_delete)} unique experiment directories ({len(orphans)} orphans).")
    
    print("\n--- Proposed Deletions ---")
    print("Directories:")
    for f in sorted(all_folders_to_delete):
        print(f"  [DELETE] {f}")
        
    confirm = input("\nDo you want to proceed with deletion? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Operation cancelled.")
        return

    # 1. Update CSV
    if rows_to_delete_indices and os.path.exists(RESULTS_CSV):
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
    for f in all_folders_to_delete:
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
