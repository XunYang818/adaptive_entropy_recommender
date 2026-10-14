$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $root
try {
    conda activate adaptive_ic
    python "src\data_preprocess.py"
    python "scripts\run_reproduction.py"
    python "scripts\run_comparison.py"
    python "scripts\run_sensitivity.py"
    python "src\plotting.py"
}
finally {
    Pop-Location
}
