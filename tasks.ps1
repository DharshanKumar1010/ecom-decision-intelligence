<#
.SYNOPSIS
    PowerShell equivalent of the Makefile targets (section 9), for native Windows use.
.EXAMPLE
    .\tasks.ps1 setup
    .\tasks.ps1 check
#>
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("setup", "data", "models", "export", "run", "app", "test", "lint", "check")]
    [string]$Task
)

$ErrorActionPreference = "Stop"
$VenvPython = ".\.venv\Scripts\python.exe"

function Invoke-Setup {
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $VenvPython -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $VenvPython -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

function Invoke-Data {
    & $VenvPython -m src.build_tables
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $VenvPython -m src.synthetic
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

function Invoke-Models {
    & $VenvPython -m src.sentiment
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $VenvPython -m src.predict
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $VenvPython -m src.ahp
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $VenvPython -m src.discovery
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

function Invoke-Export {
    & $VenvPython -m src.export_sql
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

function Invoke-Run {
    & $VenvPython run_all.py
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

function Invoke-App {
    & $VenvPython -m streamlit run app/Home.py
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

function Invoke-Test {
    & $VenvPython -m pytest -q
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

function Invoke-Lint {
    & $VenvPython -m ruff check .
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $VenvPython -m mypy src
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

function Invoke-Check {
    Invoke-Lint
    Invoke-Test
}

switch ($Task) {
    "setup"  { Invoke-Setup }
    "data"   { Invoke-Data }
    "models" { Invoke-Models }
    "export" { Invoke-Export }
    "run"    { Invoke-Run }
    "app"    { Invoke-App }
    "test"   { Invoke-Test }
    "lint"   { Invoke-Lint }
    "check"  { Invoke-Check }
}
