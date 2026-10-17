<#
.SYNOPSIS
    PowerShell equivalent of the Makefile targets (section 9), for native Windows use.
.EXAMPLE
    .\tasks.ps1 setup
    .\tasks.ps1 check
#>
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("setup", "data", "models", "export", "load-sql", "verify", "rebuild", "run", "app", "test", "lint", "check")]
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
    & $VenvPython -m src.expert
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

function Invoke-Export {
    & $VenvPython -m src.export_sql
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

function Invoke-LoadSql {
    # Loads data/export/*.csv into SQL Server (owner-authorised exception; see src/load_sql.py).
    # Extra arguments after the task name are not passed on; run python -m src.load_sql directly
    # for --dry-run, --verify-only or --only <table>.
    & $VenvPython -m src.load_sql
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

function Invoke-Verify {
    # Runs every check, prints a pass/fail table, exits non-zero if any check failed.
    # Native tools write progress to stderr; with redirected output PowerShell would turn that into
    # a terminating error under the script-wide "Stop", so judge each check by its exit code only.
    $ErrorActionPreference = "Continue"
    $steps = @(
        @{ Name = "ruff";            Args = @("-m", "ruff", "check", ".") },
        @{ Name = "mypy";            Args = @("-m", "mypy", "src") },
        @{ Name = "pytest";          Args = @("-m", "pytest", "-q") },
        @{ Name = "sql verify-only"; Args = @("-m", "src.load_sql", "--verify-only") },
        @{ Name = "RESULTS.md fresh"; Args = @("scripts/make_results.py", "--check") },
        @{ Name = "app smoke";       Args = @("scripts/app_smoke.py") }
    )
    $results = @()
    foreach ($step in $steps) {
        $clock = [Diagnostics.Stopwatch]::StartNew()
        & $VenvPython @($step.Args)
        $code = $LASTEXITCODE
        $results += [pscustomobject]@{
            Check   = $step.Name
            Status  = $(if ($code -eq 0) { "PASS" } else { "FAIL" })
            Seconds = [math]::Round($clock.Elapsed.TotalSeconds, 1)
        }
    }
    Write-Host ""
    $results | Format-Table -AutoSize | Out-String | Write-Host
    if ($results | Where-Object { $_.Status -eq "FAIL" }) { exit 1 }
}

function Invoke-Rebuild {
    # Full pipeline with the CSV export, reload SQL Server, then verify everything.
    $ErrorActionPreference = "Continue"   # see Invoke-Verify
    & $VenvPython run_all.py --export
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    Invoke-LoadSql
    Invoke-Verify
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
    "load-sql" { Invoke-LoadSql }
    "verify" { Invoke-Verify }
    "rebuild" { Invoke-Rebuild }
    "run"    { Invoke-Run }
    "app"    { Invoke-App }
    "test"   { Invoke-Test }
    "lint"   { Invoke-Lint }
    "check"  { Invoke-Check }
}
