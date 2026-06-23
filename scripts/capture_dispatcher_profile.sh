#!/usr/bin/env bash
# =============================================================================
# capture_dispatcher_profile.sh
#
# Capture one or more sequential dispatcher profiling bundles for the current
# machine.  Each run produces a timestamped directory under build/ and an
# identical copy under data/profiles/dispatcher/ (tracked in version control).
#
# USAGE
#   bash scripts/capture_dispatcher_profile.sh              # 3 runs (default)
#   bash scripts/capture_dispatcher_profile.sh --runs 1    # single run
#   bash scripts/capture_dispatcher_profile.sh --large      # extend to 2M
#   bash scripts/capture_dispatcher_profile.sh --runs 3 --large
#
# Three sequential runs are the default because the dispatch threshold
# derivation rules (see scripts/PROFILING_METHOD.md) require at least three
# runs to apply the outlier-discard and OOM-agreement checks reliably.
# Overriding with --runs 1 is useful for quick smoke tests; increase above 3
# only when bimodal instability is suspected.
#
# The --large flag extends the batch size grid from 500k to 2M elements.  Use
# it when any standard run reports V→P = 500000 (the measurement ceiling) for
# any distribution/operation.  The summarizer will print a recommendation.
#
# =============================================================================
# WINDOWS POWERSHELL TRANSLATION GUIDE
# (for AI agents producing scripts/capture_dispatcher_profile.ps1)
#
# Section mapping:
#   ARGUMENT PARSING       →  param([int]$Runs = 3, [switch]$Large)
#   PATH RESOLUTION        →  $PSScriptRoot, Join-Path, Resolve-Path
#   TOOL CHECK             →  if (-not (Test-Path $tool)) { throw "..." }
#   METADATA COLLECTION    →  see inline PS1 comments below
#   RUN LOOP               →  for ($i = 1; $i -le $Runs; $i++) { ... }
#
# Key command translations:
#   date -u +...           →  (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH-mm-ssZ")
#   uname -m               →  $env:PROCESSOR_ARCHITECTURE (or RuntimeInformation)
#   uname -s               →  "windows"
#   git rev-parse ...      →  git rev-parse --short HEAD | Out-String -NoNewline
#   awk -F= '/pattern/'    →  (Get-Content CMakeCache.txt) | Where-Object ...
#   sysctl -n ...          →  (Get-CimInstance Win32_Processor).Name
#   mkdir -p               →  New-Item -ItemType Directory -Force -Path
#   cat > file <<EOF       →  Set-Content -Path file -Value @"..."@
#   python3 script args    →  python scripts\summarize_dispatcher_profile.py $args
#   cp -R src dst          →  Copy-Item -Recurse -Path src -Destination dst
#   Tool invocation:       →  .\build\tools\strategy_profile.exe --output-csv ...
#   Strategy CSV path:     →  Join-Path $RunDir "strategy_profile_results.csv"
# =============================================================================

# ── ARGUMENT PARSING ─────────────────────────────────────────────────────────
RUNS=3
INCLUDE_LARGE=false

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --runs)
            RUNS="$2"
            shift 2
            ;;
        --large)
            INCLUDE_LARGE=true
            shift
            ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Usage: $0 [--runs N] [--large]" >&2
            exit 1
            ;;
    esac
done

if ! [[ "$RUNS" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: --runs must be a positive integer (got: '$RUNS')" >&2
    exit 1
fi

# ── PATH RESOLUTION ──────────────────────────────────────────────────────────
# PS1: $ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$PROJECT_ROOT/build}"
TOOLS_DIR="$BUILD_DIR/tools"
PROFILE_ROOT="${PROFILE_ROOT:-$BUILD_DIR/profiles/dispatcher}"
SUMMARIZER="$SCRIPT_DIR/summarize_dispatcher_profile.py"

SYSTEM_INSPECTOR="$TOOLS_DIR/system_inspector"
STRATEGY_PROFILE="$TOOLS_DIR/strategy_profile"

# ── TOOL CHECK ───────────────────────────────────────────────────────────────
# PS1: foreach ($t in $tools) { if (-not (Test-Path $t)) { throw "$t not found" } }
for tool in "$SYSTEM_INSPECTOR" "$STRATEGY_PROFILE"; do
    if [ ! -x "$tool" ]; then
        echo "Required tool not found or not executable: $tool" >&2
        echo "Build with: cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel" >&2
        exit 1
    fi
done

if [ ! -f "$SUMMARIZER" ]; then
    echo "Required summarizer not found: $SUMMARIZER" >&2
    exit 1
fi

mkdir -p "$PROFILE_ROOT"

# ── STATIC METADATA (invariant across runs within a session) ─────────────────
# PS1: $Arch  = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture
# PS1: $OsName = "windows"
# PS1: $Branch = (git rev-parse --abbrev-ref HEAD).Trim() -replace '/','-'
# PS1: $GitSha = (git rev-parse --short HEAD).Trim()
# PS1: $BuildType = (Get-Content "$BuildDir\CMakeCache.txt") | ...
# PS1: $CpuBrand = (Get-CimInstance Win32_Processor).Name
ARCH="$(uname -m)"
OS_NAME="$(uname -s | tr '[:upper:]' '[:lower:]')"
BRANCH="$(git -C "$PROJECT_ROOT" rev-parse --abbrev-ref HEAD | tr '/' '-')"
GIT_SHA="$(git -C "$PROJECT_ROOT" rev-parse --short HEAD)"
BUILD_TYPE="$(awk -F= '/^CMAKE_BUILD_TYPE:STRING=/{print $2}' "$BUILD_DIR/CMakeCache.txt" 2>/dev/null || true)"
CXX_COMPILER="$(awk -F= '/^CMAKE_CXX_COMPILER:FILEPATH=/{print $2}' "$BUILD_DIR/CMakeCache.txt" 2>/dev/null || true)"
CPU_BRAND="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "unknown")"
PHYSICAL_CORES="$(sysctl -n hw.physicalcpu 2>/dev/null || echo "unknown")"
LOGICAL_CORES="$(sysctl -n hw.logicalcpu 2>/dev/null || echo "unknown")"

LARGE_FLAG=""
if $INCLUDE_LARGE; then LARGE_FLAG="--large"; fi

echo "=================================================================="
echo " Dispatcher profile capture"
echo " Branch: $BRANCH  SHA: $GIT_SHA  Arch: $OS_NAME/$ARCH"
echo " Runs: $RUNS  Large: $INCLUDE_LARGE"
echo "=================================================================="

# ── RUN LOOP ─────────────────────────────────────────────────────────────────
# PS1: for ($i = 1; $i -le $Runs; $i++) { ...
#        $Timestamp = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH-mm-ssZ")
#        $RunId     = "${Timestamp}_windows-x86_64_${Branch}_sha-${GitSha}"
#        ... }
LAST_TRACKED_DIR=""

for (( i=1; i<=RUNS; i++ )); do
    echo ""
    echo "--- Run $i / $RUNS ---"

    # Fresh timestamp per iteration guarantees unique directory names.
    # PS1: $Timestamp = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH-mm-ssZ")
    TIMESTAMP="$(date -u +"%Y-%m-%dT%H-%M-%SZ")"
    RUN_ID="${TIMESTAMP}_${OS_NAME}-${ARCH}_${BRANCH}_sha-${GIT_SHA}"
    RUN_DIR="$PROFILE_ROOT/$RUN_ID"
    LOG_DIR="$RUN_DIR/logs"
    mkdir -p "$LOG_DIR"

    # Write metadata.json
    # PS1: Set-Content "$RunDir\metadata.json" @"{ ... }"@
    cat > "$RUN_DIR/metadata.json" <<METADATA_EOF
{
  "captured_at_utc": "$TIMESTAMP",
  "run_id": "$RUN_ID",
  "run_number": $i,
  "total_runs": $RUNS,
  "include_large": $INCLUDE_LARGE,
  "git_branch": "$BRANCH",
  "git_sha": "$GIT_SHA",
  "project_root": "$PROJECT_ROOT",
  "build_dir": "$BUILD_DIR",
  "build_type": "$BUILD_TYPE",
  "cxx_compiler": "$CXX_COMPILER",
  "os": "$OS_NAME",
  "arch": "$ARCH",
  "cpu_brand": "$CPU_BRAND",
  "physical_cores": "$PHYSICAL_CORES",
  "logical_cores": "$LOGICAL_CORES"
}
METADATA_EOF

    # Write manifest.txt
    cat > "$RUN_DIR/manifest.txt" <<MANIFEST_EOF
Dispatcher profile bundle
=========================
Run ID: $RUN_ID
Captured at (UTC): $TIMESTAMP
Run: $i of $RUNS

Files:
- metadata.json
- summary.json
- crossovers.csv          (derived: first batch where min(PARALLEL,WS) < VECTORIZED)
- best_strategies.csv     (derived: best strategy at every measured batch size)
- strategy_profile_results.csv  (canonical raw data)
- logs/system_inspector_performance.txt
- logs/strategy_profile.txt
MANIFEST_EOF

    # System capabilities snapshot
    # PS1: .\build\tools\system_inspector.exe --performance | Out-File ...
    "$SYSTEM_INSPECTOR" --performance > "$LOG_DIR/system_inspector_performance.txt" 2>&1

    # Run the forced-strategy profiler (canonical raw measurement)
    # PS1: .\build\tools\strategy_profile.exe --output-csv "$StrategyCsv" $LargeFlag
    STRATEGY_CSV="$RUN_DIR/strategy_profile_results.csv"
    # shellcheck disable=SC2086
    "$STRATEGY_PROFILE" --output-csv "$STRATEGY_CSV" $LARGE_FLAG \
        > "$LOG_DIR/strategy_profile.txt" 2>&1

    if [ ! -f "$STRATEGY_CSV" ]; then
        echo "Error: strategy_profile did not create $STRATEGY_CSV" >&2
        exit 1
    fi

    # Derive crossovers.csv, best_strategies.csv, summary.json
    # PS1: python scripts\summarize_dispatcher_profile.py "$RunDir"
    python3 "$SUMMARIZER" "$RUN_DIR"

    # Copy bundle to the version-controlled data directory
    # PS1: Copy-Item -Recurse -Path "$RunDir" -Destination "$TrackedDir"
    TRACKED_DIR="$PROJECT_ROOT/data/profiles/dispatcher/$RUN_ID"
    cp -R "$RUN_DIR" "$TRACKED_DIR"
    echo "  Tracked copy: $TRACKED_DIR"
    LAST_TRACKED_DIR="$TRACKED_DIR"
done

echo ""
echo "=================================================================="
echo " All $RUNS run(s) complete."
echo " Latest bundle: $LAST_TRACKED_DIR"
echo " Next step: apply derivation rules in scripts/PROFILING_METHOD.md"
echo "=================================================================="
