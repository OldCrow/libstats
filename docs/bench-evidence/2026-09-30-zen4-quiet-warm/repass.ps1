# Same-regime pass: corvus tools/quiet_bench.ps1 unmodified, each bench spinning 25 s in-process
# before its first measurement (LIBSTATS_BENCH_WARMUP_SECONDS), so every row is taken in the
# sustained-frequency regime this Zen 4 drops into 8-15 s after load starts.
$bench = 'C:\Users\gdwol\Development\libstats\build-bench-msvc\bench'
$out   = 'C:\Users\gdwol\Development\libstats\build-bench-msvc\quiet-2026-09-30-warm'
$env:LIBSTATS_BENCH_WARMUP_SECONDS = '25'
& 'C:\Users\gdwol\Development\corvus\tools\quiet_bench.ps1' -BuildDir $bench -OutDir $out `
    -Targets @('corvus_scaling', 'corvus_scaling_cap', 'elem_v241', 'elem_v250', 'elem_v250_cap',
               'dist_v241', 'dist_v250', 'dist_v250_cap') `
    -MaxAmbient 5 -SampleSeconds 10 -MaxWaitMinutes 30
"quiet_bench exit=$LASTEXITCODE"
