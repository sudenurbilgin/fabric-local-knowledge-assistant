$projectDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonPath = Join-Path $projectDirectory ".venv\Scripts\python.exe"
$applicationPath = Join-Path $projectDirectory "streamlit_app.py"

if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    Write-Error "The project virtual environment was not found at .venv. Create it and install requirements before launching the app."
    exit 1
}

Push-Location $projectDirectory
try {
    & $pythonPath -m streamlit run $applicationPath
    $applicationExitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}

exit $applicationExitCode
