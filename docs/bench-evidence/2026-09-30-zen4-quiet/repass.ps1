# Same-regime re-pass: corvus tools/quiet_bench.ps1 unmodified, each target listed twice so it runs
# back to back. The runner writes both runs to the same quiet_bench_<target>.txt, so the SECOND run
# (in the sustained-frequency regime, ~15 s into load on this Zen 4) is what survives; the first is the
# warm-up. The runner's 3 s noise samples between the two are too short for boost to recover.
$bench = 'C:\Users\gdwol\Development\libstats\build-bench-msvc\bench'
$out   = 'C:\Users\gdwol\Development\libstats\build-bench-msvc\quiet-2026-09-30-repass'
$t = @('corvus_scaling', 'corvus_scaling_cap', 'elem_v241', 'elem_v250', 'elem_v250_cap',
       'dist_v241', 'dist_v250', 'dist_v250_cap')
$targets = $t | ForEach-Object { $_, $_ }
& 'C:\Users\gdwol\Development\corvus\tools\quiet_bench.ps1' -BuildDir $bench -OutDir $out `
    -Targets $targets -MaxAmbient 5 -SampleSeconds 10 -MaxWaitMinutes 30
"quiet_bench exit=$LASTEXITCODE"
