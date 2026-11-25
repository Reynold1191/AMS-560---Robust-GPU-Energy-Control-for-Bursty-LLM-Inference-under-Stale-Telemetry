# run_all.ps1
# Thin wrapper around run_demo.py so there is one experiment pipeline.
# Run from an Administrator PowerShell with the project Python env active.
#
#   .\run_all.ps1              # full run: 3 seeds, all blocks (~4.5-5 h)
#   .\run_all.ps1 -Primary     # proposal run: 24 primary conditions x 3 paired seeds (~3.5-4 h)
#   .\run_all.ps1 -Demo        # 1 seed, all blocks (~1.6 h)
#   .\run_all.ps1 -QuickTest   # 1 seed, sharp_burst only (~50 min)
#   .\run_all.ps1 -SweepFrom results\attempt_4   # reuse an earlier sweep
#
# Pipeline (see run_demo.py): GPU timing check -> fixed-clock sweep ->
# SLO calibration -> oracle table -> controller matrix -> figures.

param(
    [switch]$Demo,
    [switch]$QuickTest,
    [switch]$Primary,
    [switch]$Force,
    [string]$SweepFrom = "",
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $MyInvocation.MyCommand.Path)

$argsList = @("run_demo.py")
if ($QuickTest)      { $argsList += "--quick" }
elseif ($Primary)    { $argsList += "--primary-only" }
elseif ($Demo)       { $argsList += @("--seeds", "42") }
if ($Force)          { $argsList += "--force" }
if ($SweepFrom)      { $argsList += @("--sweep-from", $SweepFrom) }

try {
    & $Python @argsList
    exit $LASTEXITCODE
}
finally {
    # Belt and braces: never leave the GPU clock locked.
    nvidia-smi -rgc | Out-Null
}
