# Z overnight quiet runs: R3 (v2.4.1 vs freeze-head costs, timing ctest), then R8
# (strategy_profile --large x3 for AVX-512, AVX2, AVX, SSE2). Writes only under $Out.
# No builds, no git writes. Keeps the system awake while it runs.
param(
    [string]$Out = "$PSScriptRoot\overnight",
    [double]$QuietPct = 8.0,        # start a run once 10-s mean CPU is below this
    [int]$QuietMaxWaitMin = 30,     # then start anyway, and log it
    [int]$WarmupSec = 25            # discarded strategy_profile pass before each R8 run
)
$ErrorActionPreference = 'Continue'
$Root = 'C:\Users\gdwol\Development\libstats-v2.4.2'
$Scratch = $PSScriptRoot
New-Item -ItemType Directory -Force $Out | Out-Null

Add-Type -Namespace Win32 -Name Power -MemberDefinition @'
[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);
'@
# ES_CONTINUOUS | ES_SYSTEM_REQUIRED: no system sleep; the display may turn off and lock.
[Win32.Power]::SetThreadExecutionState([uint32]'0x80000001') | Out-Null

function CpuPct {
    $s = Get-Counter '\Processor(_Total)\% Processor Time' -SampleInterval 2 -MaxSamples 5 -ErrorAction SilentlyContinue
    if (-not $s) { return -1 }
    [math]::Round(($s.CounterSamples | Measure-Object CookedValue -Average).Average, 1)
}
function Stamp { (Get-Date).ToString('yyyy-MM-dd HH:mm:ss') }
function UtcStamp { (Get-Date).ToUniversalTime().ToString("yyyy-MM-dd'T'HH-mm-ss'Z'") }
function Log($file, $msg) { $line = "$(Stamp) $msg"; Add-Content -Path $file -Value $line; Write-Host $line }
function AcOnline {
    try { (Get-CimInstance -Namespace root/wmi -ClassName BatteryStatus -ErrorAction Stop | Select-Object -First 1).PowerOnline } catch { 'unknown' }
}
function WaitQuiet($file, $label) {
    $t0 = Get-Date
    while ($true) {
        $c = CpuPct
        if ($c -ge 0 -and $c -lt $QuietPct) { Log $file "$label cpu=$c% ac=$(AcOnline) quiet"; return }
        if (((Get-Date) - $t0).TotalMinutes -ge $QuietMaxWaitMin) {
            $top = (Get-Process | Sort-Object CPU -Descending | Select-Object -First 5 | ForEach-Object { "$($_.ProcessName)" }) -join ','
            Log $file "$label cpu=$c% NOT quiet after $QuietMaxWaitMin min, starting anyway; top cumulative: $top"
            return
        }
        Log $file "$label cpu=$c% waiting"
        Start-Sleep -Seconds 60
    }
}

$sha = (git -C $Root rev-parse --short HEAD)
$env_log = "$Out\environment.txt"
"sha=$sha" | Set-Content $env_log
"started=$(Stamp) utc=$(UtcStamp)" | Add-Content $env_log
(powercfg /getactivescheme) | Add-Content $env_log
"ac_online=$(AcOnline)" | Add-Content $env_log
(Get-CimInstance Win32_Processor | Select-Object -First 1 | ForEach-Object { "cpu=$($_.Name) cores=$($_.NumberOfCores) logical=$($_.NumberOfLogicalProcessors)" }) | Add-Content $env_log

# ---------------- R3 ----------------
$r3 = "$Out\r3"; New-Item -ItemType Directory -Force $r3 | Out-Null
$r3log = "$r3\run_r3.txt"
WaitQuiet $r3log 'r3'
$env:LIBSTATS_BENCH_WARMUP_SECONDS = '20'
Log $r3log "start bench_v241 (LIBSTATS_BENCH_WARMUP_SECONDS=20) sha=$sha"
& "$Scratch\bench_v241.exe" > "$r3\v241.csv"
Log $r3log "end bench_v241 exit=$LASTEXITCODE cpu=$(CpuPct)%; start bench_v242"
& "$Scratch\bench_v242.exe" > "$r3\v242.csv"
Log $r3log "end bench_v242 exit=$LASTEXITCODE cpu=$(CpuPct)%"
Remove-Item Env:\LIBSTATS_BENCH_WARMUP_SECONDS
python "$Root\tools\bench\v242_compare.py" "$r3\v241.csv" "$r3\v242.csv" > "$r3\compare.txt"
Log $r3log "compare exit=$LASTEXITCODE; start timing ctest"
ctest --test-dir "$Root\build" -C Release -j1 -L timing --output-on-failure *> "$r3\timing.txt"
Log $r3log "end timing ctest exit=$LASTEXITCODE"
Log $r3log 'DONE'

# ---------------- R8 ----------------
$tiers = @(
    @{ Name = 'AVX-512'; Tag = 'avx512'; Dir = "$Root\build" },
    @{ Name = 'AVX2';    Tag = 'avx2';   Dir = "$Root\build-avx2" },
    @{ Name = 'AVX';     Tag = 'avx';    Dir = "$Root\build-avx" },
    @{ Name = 'SSE2';    Tag = 'sse2';   Dir = "$Root\build-sse2" }
)
foreach ($t in $tiers) {
    $td = "$Out\r8_$($t.Tag)"; New-Item -ItemType Directory -Force "$td\logs" | Out-Null
    $runlog = "$td\logs\run.txt"
    $tools = "$($t.Dir)\tools\Release"
    $prof = "$tools\strategy_profile.exe"
    Push-Location $tools
    $si = (& "$tools\system_inspector.exe" --quick 2>&1)
    Pop-Location
    $si | Set-Content "$td\logs\system_inspector.txt"
    Log $runlog "batch start tier=$($t.Name) build=$($t.Dir) sha=$sha runs=1 2 3"
    Log $runlog "system_inspector: $(($si | Select-String 'SIMD' | Select-Object -First 1))"
    for ($i = 1; $i -le 3; $i++) {
        WaitQuiet $runlog "run$i"
        Push-Location $td
        $w = Start-Process -FilePath $prof -WorkingDirectory $td -PassThru -WindowStyle Hidden `
             -RedirectStandardOutput "$td\logs\warmup$i.out" -ArgumentList '-o', "$td\warmup$i.csv"
        Start-Sleep -Seconds $WarmupSec
        if (-not $w.HasExited) { Stop-Process -Id $w.Id -Force }
        Remove-Item "$td\warmup$i.csv", "$td\logs\warmup$i.out" -ErrorAction SilentlyContinue
        if ($i -eq 1) { $t.Utc = UtcStamp }
        Log $runlog "run$i start (after ${WarmupSec}s discarded strategy_profile warm-up) utc=$(UtcStamp)"
        & $prof --large -o "$td\strategy_profile_run$i.csv" *> "$td\logs\profile_run$i.out"
        $rc = $LASTEXITCODE
        $rows = if (Test-Path "$td\strategy_profile_run$i.csv") { (Get-Content "$td\strategy_profile_run$i.csv").Count } else { 0 }
        Pop-Location
        Log $runlog "run$i end exit=$rc rows=$rows cpu=$(CpuPct)%"
    }
    "run1_utc=$($t.Utc)" | Set-Content "$td\run1_utc.txt"
    Log $runlog 'batch DONE'
}
"finished=$(Stamp)" | Add-Content $env_log
[Win32.Power]::SetThreadExecutionState([uint32]'0x80000000') | Out-Null
