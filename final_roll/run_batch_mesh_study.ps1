# Runner PowerShell per lo studio di convergenza mesh
# Esegue in sequenza 12k, 29k e 52k con il virtual environment corretto

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  AVVIO BATCH RUNNER - STUDIO CONVERGENZA MESH (12k, 29k, 52k)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

$VENV_PYTHON = "..\venv\Scripts\python.exe"
if (-not (Test-Path $VENV_PYTHON)) {
    $VENV_PYTHON = "venv\Scripts\python.exe"
}
if (-not (Test-Path $VENV_PYTHON)) {
    $VENV_PYTHON = "python.exe"
}

& $VENV_PYTHON run_batch_mesh_study.py @args
