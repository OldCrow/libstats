# =============================================================================
# capture_dispatcher_profile.ps1
#
# Windows PowerShell port of capture_dispatcher_profile.sh.
# Captures sequential dispatcher profiling bundles for the current machine.
# Each run produces a timestamped directory under build/profiles/dispatcher/
# and an identical copy under data/profiles/dispatcher/ (tracked in VCS).
#
# USAGE
#   .\scripts\capture_dispatcher_profile.ps1              # 3 runs (default)
#   .\scripts\capture_dispatcher_profile.ps1 -Runs 1      # single smoke test
#   .\scripts\capture_dispatcher_profile.ps1 -Large       # extend to 2M
#   .\scripts\capture_dispatcher_profile.ps1 -Runs 3 -Large
#
# See scripts/PROFILING_METHOD.md for threshold derivation rules.
# =============================================================================
[CmdletBinding()]
param(
    [int]   $Runs  = 3,
    [switch]$Large
)

Set-StrictMode -Off
$ErrorActionPreference = 'Stop'
trap { Write-Error "TRAP line $($_.InvocationInfo.ScriptLineNumber): $_"; break }

# ── PATH RESOLUTION ───────────────────────────────────────────────────────────
$ProjectRoot  = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$BuildDir     = Join-Path $ProjectRoot 'build'
$ToolsDir     = Join-Path $BuildDir    'tools\Release'
$ProfileRoot  = Join-Path $BuildDir    'profiles\dispatcher'
$Summarizer   = Join-Path $PSScriptRoot 'summarize_dispatcher_profile.py'

$SystemInspector  = Join-Path $ToolsDir 'system_inspector.exe'
$StrategyProfile  = Join-Path $ToolsDir 'strategy_profile.exe'

# ── TOOL CHECK ────────────────────────────────────────────────────────────────
foreach ($t in @($SystemInspector, $StrategyProfile)) {
    if (-not (Test-Path $t)) {
        throw "Required tool not found: $t`nBuild with: cmake --build build --config Release --parallel"
    }
}
if (-not (Test-Path $Summarizer)) {
    throw "Required summarizer not found: $Summarizer"
}

New-Item -ItemType Directory -Force -Path $ProfileRoot | Out-Null

# ── STATIC METADATA (invariant across runs) ───────────────────────────────────
# Map PROCESSOR_ARCHITECTURE to uname-m style names (matches bash script convention)
$Arch = switch ($env:PROCESSOR_ARCHITECTURE) {
    'AMD64' { 'x86_64' }
    'ARM64' { 'arm64'  }
    'x86'   { 'x86'    }
    default { $env:PROCESSOR_ARCHITECTURE.ToLower() }
}
$OsName   = 'windows'
$Branch   = ([string](git -C $ProjectRoot rev-parse --abbrev-ref HEAD)).Trim() -replace '/', '-'
$GitSha   = ([string](git -C $ProjectRoot rev-parse --short HEAD)).Trim()

$CacheFile = Join-Path $BuildDir 'CMakeCache.txt'
$BuildType = (Get-Content $CacheFile -ErrorAction SilentlyContinue) |
    Where-Object { $_ -match '^CMAKE_BUILD_TYPE:STRING=(.*)' } |
    ForEach-Object { $Matches[1] } | Select-Object -First 1
if (-not $BuildType) { $BuildType = 'Release' }   # VS generator: type set at build time

$CxxCompiler = (Get-Content $CacheFile -ErrorAction SilentlyContinue) |
    Where-Object { $_ -match '^CMAKE_CXX_COMPILER:FILEPATH=(.*)' } |
    ForEach-Object { $Matches[1] } | Select-Object -First 1
if (-not $CxxCompiler) { $CxxCompiler = 'unknown' }

$cpu_inst      = Get-CimInstance Win32_Processor | Select-Object -First 1
$CpuBrand      = [string]$cpu_inst.Name
$PhysicalCores = [int]$cpu_inst.NumberOfCores
$LogicalCores  = [int]$cpu_inst.NumberOfLogicalProcessors


Write-Host "=================================================================="
Write-Host "  Dispatcher profile capture"
Write-Host "  Branch: $Branch  SHA: $GitSha  Arch: $OsName/$Arch"
Write-Host "  Runs: $Runs  Large: $($Large.IsPresent)"
Write-Host "=================================================================="

# ── RUN LOOP ──────────────────────────────────────────────────────────────────
$LastTrackedDir = ''

for ($i = 1; $i -le $Runs; $i++) {
    Write-Host ""
    Write-Host "--- Run $i / $Runs ---"

    $Timestamp = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH-mm-ssZ')
    $RunId     = "${Timestamp}_${OsName}-${Arch}_${Branch}_sha-${GitSha}"
    $RunDir    = Join-Path $ProfileRoot $RunId
    $LogDir    = Join-Path $RunDir 'logs'
    New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

    # Write metadata.json
    $IncludeLargeStr = if ($Large) { 'true' } else { 'false' }
    $MetadataJson = @"
{
  "captured_at_utc": "$Timestamp",
  "run_id": "$RunId",
  "run_number": $i,
  "total_runs": $Runs,
  "include_large": $IncludeLargeStr,
  "git_branch": "$Branch",
  "git_sha": "$GitSha",
  "project_root": "$($ProjectRoot -replace '\\','\\')",
  "build_dir": "$($BuildDir -replace '\\','\\')",
  "build_type": "Release",
  "cxx_compiler": "$($CxxCompiler -replace '\\','\\')",
  "os": "$OsName",
  "arch": "$Arch",
  "cpu_brand": "$CpuBrand",
  "physical_cores": $PhysicalCores,
  "logical_cores": $LogicalCores
}
"@
    Set-Content -Path (Join-Path $RunDir 'metadata.json') -Value $MetadataJson -Encoding utf8NoBOM

    # Write manifest.txt
    $ManifestTxt = @"
Dispatcher profile bundle
=========================
Run ID: $RunId
Captured at (UTC): $Timestamp
Run: $i of $Runs

Files:
- metadata.json
- summary.json
- crossovers.csv          (derived: first batch where min(PARALLEL,WS) < VECTORIZED)
- best_strategies.csv     (derived: best strategy at every measured batch size)
- strategy_profile_results.csv  (canonical raw data)
- logs/system_inspector_performance.txt
- logs/strategy_profile.txt
"@
    Set-Content -Path (Join-Path $RunDir 'manifest.txt') -Value $ManifestTxt -Encoding utf8NoBOM

    # System capabilities snapshot
    Write-Host "  Running system_inspector --performance..."
    & $SystemInspector --performance | Out-File -FilePath (Join-Path $LogDir 'system_inspector_performance.txt') -Encoding UTF8

    # Run the forced-strategy profiler
    $StrategyCsv = Join-Path $RunDir 'strategy_profile_results.csv'
    Write-Host "  Running strategy_profile --output-csv..."
    $ProfileArgs = @('--output-csv', $StrategyCsv)
    if ($Large) { $ProfileArgs += '--large' }
    & $StrategyProfile @ProfileArgs | Out-File -FilePath (Join-Path $LogDir 'strategy_profile.txt') -Encoding UTF8

    if (-not (Test-Path $StrategyCsv)) {
        throw "strategy_profile did not create $StrategyCsv"
    }

    # Derive crossovers.csv, best_strategies.csv, summary.json
    Write-Host "  Running summarizer..."
    python (Join-Path $PSScriptRoot 'summarize_dispatcher_profile.py') $RunDir

    # Copy bundle to version-controlled data directory
    $TrackedDir = Join-Path $ProjectRoot "data\profiles\dispatcher\$RunId"
    Copy-Item -Recurse -Path $RunDir -Destination $TrackedDir -Force
    Write-Host "  Tracked copy: $TrackedDir"
    $LastTrackedDir = $TrackedDir
}

Write-Host ""
Write-Host "=================================================================="
Write-Host "  All $Runs run(s) complete."
Write-Host "  Latest bundle: $LastTrackedDir"
Write-Host "  Next step: apply derivation rules in scripts/PROFILING_METHOD.md"
Write-Host "=================================================================="
