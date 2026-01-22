# Specification - Heat2D Results and Experiments Cleanup

## Overview
This track involves creating a maintenance script to clean up the `Heat2D` experiment data. The script will filter the `results.csv` file and remove entries with fewer than 10,000 epochs, subsequently deleting the associated experiment folders.

## Functional Requirements
- **CSV Filtering:** Read `Heat2D/results.csv` and identify all rows where the `Epochs` column value is less than 10,000.
- **Directory Mapping:** For each identified row, locate the corresponding experiment directory within `Heat2D/experiments/`.
    - Logic: Search for parent directories (e.g., `L2_50x4_1_E100_GELU`) matching the architecture/epochs/activation and subdirectories (e.g., `0_NN_Random`) matching the run type.
- **Dry Run Mode:** By default, the script must list all CSV rows and filesystem directories targeted for removal and require explicit user confirmation (y/n) before performing any deletions.
- **Cleanup Action:** 
    - Remove identified rows from `results.csv`.
    - Recursively delete identified experiment directories.

## Non-Functional Requirements
- **Safety:** Ensure the script handles missing directories gracefully (e.g., if a CSV entry exists but the folder was already deleted).
- **Fixed Threshold:** The 10,000 epoch limit is hardcoded.

## Acceptance Criteria
- [ ] Script successfully identifies rows with < 10,000 epochs in `Heat2D/results.csv`.
- [ ] Script correctly maps these rows to their respective folders in `Heat2D/experiments/`.
- [ ] Script displays a clear list of "to be deleted" items and waits for confirmation.
- [ ] Upon confirmation, the CSV is updated and folders are removed.
- [ ] After execution, no entries with < 10,000 epochs remain in the CSV for the Heat2D case.

## Out of Scope
- Cleaning up results for cases other than Heat2D (e.g., CSTR, Harmonic Oscillator).
- Automated backups before deletion (user opted for Dry Run confirmation instead).
