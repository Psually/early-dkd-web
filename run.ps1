$ErrorActionPreference = 'Stop'
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$localPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$python = if (Test-Path -LiteralPath $localPython) { $localPython } else { 'python' }
Write-Host 'Open http://127.0.0.1:8765/ . Press Ctrl+C to stop.'
& $python (Join-Path $PSScriptRoot 'serve.py')
