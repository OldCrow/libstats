# Table A/B on Z: for each tier, old, new, old, new back to back (ABAB), every bench process opted out of
# power throttling, a discarded warm-up pass first, each tier gated on CPU < $QuietPct (bounded wait).
# Writes <Out>/<tier>/tab_{old,new}_{1,2}.csv and run.txt. No builds, no git writes.
param(
    [string]$Out = "$PSScriptRoot\ab_round$Round",
    [int]$Round = 1,
    [string[]]$Tiers = @('avx512', 'avx2', 'avx', 'sse2'),
    [double]$QuietPct = 8.0,
    [int]$QuietMaxWaitMin = 10,
    [int]$WarmupSec = 20
)
$ErrorActionPreference = 'Continue'
$Scratch = $PSScriptRoot
if (-not $PSBoundParameters.ContainsKey('Out')) { $Out = "$Scratch\ab_round$Round" }
New-Item -ItemType Directory -Force $Out | Out-Null
. "$Scratch\nothrottle.ps1"
Add-Type -Namespace Win32 -Name Power -MemberDefinition @'
[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);
'@
[Win32.Power]::SetThreadExecutionState([uint32]'0x80000001') | Out-Null
function CpuPct {
    $s = Get-Counter '\Processor(_Total)\% Processor Time' -SampleInterval 2 -MaxSamples 3 -ErrorAction SilentlyContinue
    if (-not $s) { return -1 }
    [math]::Round(($s.CounterSamples | Measure-Object CookedValue -Average).Average, 1)
}
function Stamp { (Get-Date).ToString('yyyy-MM-dd HH:mm:ss') }
function Log($file, $msg) { $line = "$(Stamp) $msg"; Add-Content -Path $file -Value $line; Write-Host $line }
function WaitQuiet($file, $label) {
    $t0 = Get-Date
    while ($true) {
        $c = CpuPct
        if ($c -ge 0 -and $c -lt $QuietPct) { Log $file "$label cpu=$c% quiet"; return }
        if (((Get-Date) - $t0).TotalMinutes -ge $QuietMaxWaitMin) { Log $file "$label cpu=$c% NOT quiet after $QuietMaxWaitMin min, starting anyway (indicative)"; return }
        Log $file "$label cpu=$c% waiting"; Start-Sleep -Seconds 30
    }
}
function StartOptedOut($exe, $wd, $stdout, $stderr) {
    $p = Start-Process -FilePath $exe -WorkingDirectory $wd -PassThru -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    $ok = Disable-PowerThrottling $p
    $p | Add-Member -NotePropertyName OptOut -NotePropertyValue "set=$ok $(Get-PowerThrottling $p)"
    return $p
}
function RunOptedOut($file, $label, $exe, $wd, $stdout) {
    $p = StartOptedOut $exe $wd $stdout "$stdout.err"
    Log $file "$label start ($($p.OptOut))"
    $t0 = Get-Date; $p.WaitForExit()
    if ((Test-Path "$stdout.err") -and (Get-Item "$stdout.err").Length -eq 0) { Remove-Item "$stdout.err" }
    $rows = if (Test-Path $stdout) { (Get-Content $stdout).Count } else { 0 }
    Log $file "$label end exit=$($p.ExitCode) rows=$rows $([int]((Get-Date)-$t0).TotalSeconds)s cpu=$(CpuPct)%"
}
function ClockCheck($file, $label) {
    $o = "$Out\clock_$label.txt"
    RunOptedOut $file "clock[$label]" "$Scratch\prev\probe2.exe" $Out $o | Out-Null
    $vals = Get-Content $o | ForEach-Object { if ($_ -match '([\d.]+) ms') { [double]$Matches[1] } }
    Log $file "clock[$label] ms/pass=$($vals -join '/') (boost ~128, base ~190)"
}
"round=$Round started=$(Stamp) head=$(git -C 'C:\Users\gdwol\Development\libstats-v2.4.2' rev-parse --short HEAD) old=$(git -C 'C:\Users\gdwol\Development\libstats-oldtable' rev-parse --short HEAD)" | Set-Content "$Out\environment.txt"
foreach ($tier in $Tiers) {
    $d = "$Out\$tier"; New-Item -ItemType Directory -Force $d | Out-Null
    $log = "$d\run.txt"
    WaitQuiet $log $tier
    ClockCheck $log $tier
    # discarded warm-up: the old bench, killed after $WarmupSec
    $w = StartOptedOut "$Scratch\table_ab_old_$tier.exe" $d "$d\warmup.csv" "$d\warmup.err"
    Start-Sleep -Seconds $WarmupSec
    if (-not $w.HasExited) { Stop-Process -Id $w.Id -Force }
    Start-Sleep -Milliseconds 500
    Remove-Item "$d\warmup.csv", "$d\warmup.err" -ErrorAction SilentlyContinue
    Log $log "$tier warm-up ${WarmupSec}s discarded"
    foreach ($pass in 1, 2) {
        foreach ($side in 'old', 'new') {
            RunOptedOut $log "$tier $side pass $pass" "$Scratch\table_ab_${side}_$tier.exe" $d "$d\tab_${side}_$pass.csv"
        }
    }
    Log $log "$tier DONE"
}
"finished=$(Stamp)" | Add-Content "$Out\environment.txt"
[Win32.Power]::SetThreadExecutionState([uint32]'0x80000000') | Out-Null
