# Z quiet runs on the clang-cl builds at d1adc95 (branch tip df78931), 2026-10-13/14: the speed gate
# alone, the R3 cost benches (v2.4.1 vs head) and compare, the timing tests one by one, then R8
# (strategy_profile --large x3 for AVX-512, AVX2, AVX, SSE2). Every measured process is opted out of
# Windows power throttling (a locked screen otherwise drops it to base clock, and children do not
# inherit the opt-out), so each binary is launched here, not through ctest.
# Writes only under $Out. No builds, no git writes. Keeps the system awake while it runs.
param(
    [string]$Out = "$PSScriptRoot\overnight4",
    [double]$QuietPct = 8.0,
    [int]$QuietMaxWaitMin = 30,
    [int]$WarmupSec = 25,
    [switch]$SelfTest
)
$ErrorActionPreference = 'Continue'
$Root = 'C:\Users\gdwol\Development\libstats-v2.4.2'
$Scratch = $PSScriptRoot
New-Item -ItemType Directory -Force $Out | Out-Null
. "$Scratch\nothrottle.ps1"

Add-Type -Namespace Win32 -Name Power -MemberDefinition @'
[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);
'@
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
            Log $file "$label cpu=$c% NOT quiet after $QuietMaxWaitMin min, starting anyway"
            return
        }
        Log $file "$label cpu=$c% waiting"
        Start-Sleep -Seconds 60
    }
}
# Start a process, opt it out of power throttling at once, and return it.
function StartOptedOut($exe, [string[]]$argv, $wd, $stdout, $stderr) {
    $sp = @{ FilePath = $exe; WorkingDirectory = $wd; PassThru = $true; WindowStyle = 'Hidden';
             RedirectStandardOutput = $stdout; RedirectStandardError = $stderr }
    if ($argv -and $argv.Count) { $sp.ArgumentList = $argv }
    $p = Start-Process @sp
    $ok = Disable-PowerThrottling $p
    $p | Add-Member -NotePropertyName OptOut -NotePropertyValue "set=$ok $(Get-PowerThrottling $p)"
    return $p
}
function RunOptedOut($file, $label, $exe, [string[]]$argv, $wd, $stdout) {
    $p = StartOptedOut $exe $argv $wd $stdout "$stdout.err"
    Log $file "$label start ($($p.OptOut))"
    $p.WaitForExit()
    $rc = $p.ExitCode
    if ((Test-Path "$stdout.err") -and (Get-Item "$stdout.err").Length -eq 0) { Remove-Item "$stdout.err" }
    return $rc
}
# 10-s single-thread clock check, opted out: ~128 ms per pass at boost, ~190 at base clock.
function ClockCheck($file, $label) {
    $o = "$Out\clock_$label.txt"
    $rc = RunOptedOut $file "clock[$label]" "$Scratch\prev\probe2.exe" @('10') $Out $o
    $vals = Get-Content $o | ForEach-Object { if ($_ -match '([\d.]+) ms') { [double]$Matches[1] } }
    Log $file "clock[$label] rc=$rc ms/pass=$($vals -join '/') (boost ~128, base ~190)"
}

# The timing tests of the clang-cl tree, as ctest would launch them (regenerated here, after the build).
$timingJson = "$Out\timing_tests.json"
ctest --test-dir "$Root\build-clangcl" --show-only=json-v1 -L timing 2>$null | Set-Content $timingJson
$tests = (Get-Content $timingJson -Raw | ConvertFrom-Json).tests

if ($SelfTest) {
    $log = "$Out\selftest.txt"
    Log $log "timing tests: $($tests.Count): $(($tests | ForEach-Object name) -join ', ')"
    ClockCheck $log 'selftest'
    $t = $tests[0]
    $wd = ($t.properties | Where-Object name -eq 'WORKING_DIRECTORY').value
    $rc = RunOptedOut $log "  $($t.name)" $t.command[0] @() $wd "$Out\selftest_$($t.name).txt"
    Log $log "  $($t.name) exit=$rc"
    $w = StartOptedOut "$Root\build-clangcl\tools\strategy_profile.exe" @('-o', "$Out\selftest_warm.csv") $Out "$Out\selftest_warm.out" "$Out\selftest_warm.err"
    Start-Sleep -Seconds 3
    if (-not $w.HasExited) { Stop-Process -Id $w.Id -Force }
    Log $log "warm-up start/kill ok ($($w.OptOut))"
    foreach ($b in 'bench_cc_v241', 'bench_cc_v242') { Log $log "$b exists=$(Test-Path "$Scratch\$b.exe")" }
    [Win32.Power]::SetThreadExecutionState([uint32]'0x80000000') | Out-Null
    return
}

$sha = (git -C $Root rev-parse --short HEAD)
$env_log = "$Out\environment.txt"
"sha=$sha code_sha=d1adc95" | Set-Content $env_log
"started=$(Stamp) utc=$(UtcStamp)" | Add-Content $env_log
(powercfg /getactivescheme) | Add-Content $env_log
"ac_online=$(AcOnline)" | Add-Content $env_log
(Get-CimInstance Win32_Processor | Select-Object -First 1 | ForEach-Object { "cpu=$($_.Name) cores=$($_.NumberOfCores) logical=$($_.NumberOfLogicalProcessors)" }) | Add-Content $env_log
"os=$((Get-CimInstance Win32_OperatingSystem).Caption) $((Get-CimInstance Win32_OperatingSystem).Version)" | Add-Content $env_log
"compiler=clang-cl 22.1.3 (VS 2026 18.11 bundled), Ninja, Release" | Add-Content $env_log

# ---------------- the speed gate alone ----------------
$g = "$Out\gates"; New-Item -ItemType Directory -Force $g | Out-Null
$glog = "$g\run_gates.txt"
WaitQuiet $glog 'gates'
ClockCheck $glog 'gates'
$gate = $tests | Where-Object name -eq 'test_parallel_batch_gates'
$wd = ($gate.properties | Where-Object name -eq 'WORKING_DIRECTORY').value
$rc = RunOptedOut $glog 'test_parallel_batch_gates' $gate.command[0] @() $wd "$g\test_parallel_batch_gates.txt"
Log $glog "test_parallel_batch_gates exit=$rc"

# ---------------- R3: costs against v2.4.1 ----------------
$r3 = "$Out\r3"; New-Item -ItemType Directory -Force $r3 | Out-Null
$r3log = "$r3\run_r3.txt"
WaitQuiet $r3log 'r3'
ClockCheck $r3log 'r3'
$env:LIBSTATS_BENCH_WARMUP_SECONDS = '20'
$rc = RunOptedOut $r3log "bench_cc_v241 (WARMUP 20) sha=$sha" "$Scratch\bench_cc_v241.exe" @() $r3 "$r3\v241.csv"
Log $r3log "end bench_cc_v241 exit=$rc cpu=$(CpuPct)%"
$rc = RunOptedOut $r3log 'bench_cc_v242 (WARMUP 20)' "$Scratch\bench_cc_v242.exe" @() $r3 "$r3\v242.csv"
Log $r3log "end bench_cc_v242 exit=$rc cpu=$(CpuPct)%"
Remove-Item Env:\LIBSTATS_BENCH_WARMUP_SECONDS
python -I "$Root\tools\bench\v242_compare.py" "$r3\v241.csv" "$r3\v242.csv" > "$r3\compare.txt"
Log $r3log "compare exit=$LASTEXITCODE"

# ---------------- the timing tests, one by one ----------------
$td = "$r3\timing"; New-Item -ItemType Directory -Force $td | Out-Null
WaitQuiet $r3log 'timing'
ClockCheck $r3log 'timing'
$pass = 0; $fail = @()
foreach ($t in $tests) {
    $wd = ($t.properties | Where-Object name -eq 'WORKING_DIRECTORY').value
    $t0 = Get-Date
    $rc = RunOptedOut $r3log "  $($t.name)" $t.command[0] @() $wd "$td\$($t.name).txt"
    $secs = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)
    if ($rc -eq 0) { $pass++ } else { $fail += $t.name }
    Log $r3log "  $($t.name) exit=$rc ${secs}s"
}
"timing: $pass/$($tests.Count) passed; failed: $($fail -join ', ')" | Set-Content "$r3\timing.txt"
Log $r3log "timing: $pass/$($tests.Count) passed $($fail -join ', ')"
Log $r3log 'DONE'

# ---------------- R8 ----------------
$tiers = @(
    @{ Name = 'AVX-512'; Tag = 'avx512'; Dir = "$Root\build-clangcl" },
    @{ Name = 'AVX2';    Tag = 'avx2';   Dir = "$Root\build-clangcl-avx2" },
    @{ Name = 'AVX';     Tag = 'avx';    Dir = "$Root\build-clangcl-avx" },
    @{ Name = 'SSE2';    Tag = 'sse2';   Dir = "$Root\build-clangcl-sse2" }
)
foreach ($t in $tiers) {
    $td = "$Out\r8_$($t.Tag)"; New-Item -ItemType Directory -Force "$td\logs" | Out-Null
    $runlog = "$td\logs\run.txt"
    $tools = "$($t.Dir)\tools"
    $prof = "$tools\strategy_profile.exe"
    $si = (& "$tools\system_inspector.exe" --quick 2>&1)
    $si | Set-Content "$td\logs\system_inspector.txt"
    Log $runlog "batch start tier=$($t.Name) build=$($t.Dir) sha=$sha runs=1 2 3"
    Log $runlog "system_inspector: $(($si | Select-String 'System:' | Select-Object -First 1))"
    for ($i = 1; $i -le 3; $i++) {
        WaitQuiet $runlog "run$i"
        ClockCheck $runlog "$($t.Tag)_run$i"
        $w = StartOptedOut $prof @('-o', "$td\warmup$i.csv") $td "$td\logs\warmup$i.out" "$td\logs\warmup$i.err"
        Start-Sleep -Seconds $WarmupSec
        if (-not $w.HasExited) { Stop-Process -Id $w.Id -Force }
        Start-Sleep -Milliseconds 500
        Remove-Item "$td\warmup$i.csv", "$td\logs\warmup$i.out", "$td\logs\warmup$i.err" -ErrorAction SilentlyContinue
        if ($i -eq 1) { $t.Utc = UtcStamp }
        Log $runlog "run$i (after ${WarmupSec}s discarded opted-out strategy_profile warm-up) utc=$(UtcStamp)"
        $rc = RunOptedOut $runlog "run$i profile" $prof @('--large', '-o', "$td\strategy_profile_run$i.csv") $td "$td\logs\profile_run$i.out"
        $rows = if (Test-Path "$td\strategy_profile_run$i.csv") { (Get-Content "$td\strategy_profile_run$i.csv").Count } else { 0 }
        Log $runlog "run$i end exit=$rc rows=$rows cpu=$(CpuPct)%"
    }
    "run1_utc=$($t.Utc)" | Set-Content "$td\run1_utc.txt"
    Log $runlog 'batch DONE'
}
"finished=$(Stamp)" | Add-Content $env_log
[Win32.Power]::SetThreadExecutionState([uint32]'0x80000000') | Out-Null
