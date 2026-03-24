# Run the Medical AI Detection System (Streamlit)
# Usage: .\run.ps1   or   pwsh -File run.ps1

Set-Location $PSScriptRoot

if (-not (Test-Path "venv")) {
    Write-Host "Creating virtual environment..."
    python -m venv venv
}
& .\venv\Scripts\Activate.ps1
pip install -r requirements.txt -q
Write-Host "Starting Streamlit app..."
streamlit run app.py
