#!/bin/bash

# Capture a dispatcher profiling bundle for the current machine.
# Saves metadata, logs, and benchmark CSV output in a timestamped directory under build/.
# Copies the bundle into data/profiles/dispatcher/ (tracked in version control) so
# profiles from all architectures can be consolidated on any machine.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$PROJECT_ROOT/build}"
TOOLS_DIR="$BUILD_DIR/tools"
PROFILE_ROOT="${PROFILE_ROOT:-$BUILD_DIR/profiles/dispatcher}"
SUMMARIZER="$SCRIPT_DIR/summarize_dispatcher_profile.py"

SYSTEM_INSPECTOR="$TOOLS_DIR/system_inspector"
STRATEGY_PROFILE="$TOOLS_DIR/strategy_profile"

for tool in "$SYSTEM_INSPECTOR" "$STRATEGY_PROFILE"; do
    if [ ! -x "$tool" ]; then
        echo "Required tool not found or not executable: $tool" >&2
        exit 1
    fi
done

if [ ! -f "$SUMMARIZER" ]; then
    echo "Required summarizer not found: $SUMMARIZER" >&2
    exit 1
fi

mkdir -p "$PROFILE_ROOT"

TIMESTAMP="$(date -u +"%Y-%m-%dT%H-%M-%SZ")"
ARCH="$(uname -m)"
OS_NAME="$(uname -s | tr '[:upper:]' '[:lower:]')"
BRANCH="$(git -C "$PROJECT_ROOT" rev-parse --abbrev-ref HEAD)"
GIT_SHA="$(git -C "$PROJECT_ROOT" rev-parse --short HEAD)"
RUN_ID="${TIMESTAMP}_${OS_NAME}-${ARCH}_${BRANCH}_sha-${GIT_SHA}"
RUN_DIR="$PROFILE_ROOT/$RUN_ID"
LOG_DIR="$RUN_DIR/logs"

mkdir -p "$LOG_DIR"

BUILD_TYPE="$(awk -F= '/^CMAKE_BUILD_TYPE:STRING=/{print $2}' "$BUILD_DIR/CMakeCache.txt" 2>/dev/null || true)"
CXX_COMPILER="$(awk -F= '/^CMAKE_CXX_COMPILER:FILEPATH=/{print $2}' "$BUILD_DIR/CMakeCache.txt" 2>/dev/null || true)"
CPU_BRAND="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "unknown")"
PHYSICAL_CORES="$(sysctl -n hw.physicalcpu 2>/dev/null || echo "unknown")"
LOGICAL_CORES="$(sysctl -n hw.logicalcpu 2>/dev/null || echo "unknown")"

cat > "$RUN_DIR/metadata.json" <<EOF
{
  "captured_at_utc": "$TIMESTAMP",
  "run_id": "$RUN_ID",
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
EOF

cat > "$RUN_DIR/manifest.txt" <<EOF
Dispatcher profile bundle
=========================

Run ID: $RUN_ID
Captured at (UTC): $TIMESTAMP

Files:
- metadata.json
- summary.json
- crossovers.csv
- best_strategies.csv
- strategy_profile_results.csv
- logs/system_inspector_performance.txt
- logs/strategy_profile.txt
EOF

echo "Capturing dispatcher profile to: $RUN_DIR"

"$SYSTEM_INSPECTOR" --performance > "$LOG_DIR/system_inspector_performance.txt" 2>&1

STRATEGY_CSV="$RUN_DIR/strategy_profile_results.csv"
"$STRATEGY_PROFILE" --output-csv "$STRATEGY_CSV" > "$LOG_DIR/strategy_profile.txt" 2>&1

if [ ! -f "$STRATEGY_CSV" ]; then
    echo "Expected strategy profile CSV was not created." >&2
    exit 1
fi

python3 "$SUMMARIZER" "$RUN_DIR"

# Copy bundle into the tracked data directory so profiles accumulate across machines.
TRACKED_DIR="$PROJECT_ROOT/data/profiles/dispatcher/$RUN_ID"
cp -R "$RUN_DIR" "$TRACKED_DIR"
echo "Dispatcher profile saved to: $RUN_DIR"
echo "Tracked copy at: $TRACKED_DIR"
