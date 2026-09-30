# Compile tools/bench/*.cpp against the v2.4.1 worktree (OLD) and the branch (NEW), MSVC, Release.
# Bench TUs take the same /arch flag libstats applies globally on this machine.
$ErrorActionPreference = 'Stop'

$vcvars = 'C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat'
cmd /c "`"$vcvars`" > nul && set" | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') { Set-Item -Path "env:$($Matches[1])" -Value $Matches[2] }
}

$new    = 'C:\Users\gdwol\Development\libstats'
$old    = 'C:\Users\gdwol\Development\libstats-v2.4.1'
$prefix = "$new\build-bench-deps\prefix"
$src    = "$new\tools\bench"
$out    = "$new\build-bench-msvc\bench"
New-Item -ItemType Directory -Force $out | Out-Null
Set-Location $out

# Flags and definitions copied from the library's own TUs (build.ninja), so headers see the same
# configuration on both sides of the link.
$cxx     = @('/nologo', '/std:c++20', '/O2', '/Ob2', '/DNDEBUG', '/EHsc', '/MD', '/utf-8', '/arch:AVX512',
             '/DNOMINMAX', '/D_USE_MATH_DEFINES', '/D_CRT_SECURE_NO_WARNINGS', '/D_CRT_NONSTDC_NO_DEPRECATE',
             '/DLIBSTATS_HAS_SSE2=1', '/DLIBSTATS_HAS_AVX=1', '/DLIBSTATS_HAS_AVX2=1', '/DLIBSTATS_HAS_AVX512=1')
$corvus  = @("$prefix\lib\corvus.lib", "$prefix\lib\hwy.lib", "$prefix\lib\hwy_contrib.lib")

function Build($name, $source, $incs, $libs) {
    $args = $cxx + ($incs | ForEach-Object { "/I$_" }) + @("$src\$source", "/Fe:$name.exe", "/Fo:$name.obj", '/link') + $libs
    & cl @args
    if ($LASTEXITCODE -ne 0) { throw "$name failed" }
}

$oldInc = @("$old\include", "$old\include\libstats", "$old\build-bench-msvc\generated")
$newInc = @("$new\include", "$new\include\libstats", "$new\build-bench-msvc\generated")
$oldLib = @("$old\build-bench-msvc\stats_static.lib")
$newLib = @("$new\build-bench-msvc\stats_static.lib") + $corvus

Build 'dist_v241'  'distributions_bench.cpp'  $oldInc $oldLib
Build 'dist_v250'  'distributions_bench.cpp'  $newInc $newLib
Build 'elem_v241'  'elementary_bench.cpp'     $oldInc $oldLib
Build 'elem_v250'  'elementary_bench.cpp'     $newInc $newLib
Build 'corvus_scaling' 'corvus_scaling_bench.cpp' @("$prefix\include") $corvus

# All-MSVC leg: corvus + Highway built by cl.exe, so corvus stops at Highway's MSVC AVX2 cap.
# What a default FetchContent build of the branch produces on Windows.
$capPrefix = "$new\build-bench-deps\prefix-msvc"
if (Test-Path "$capPrefix\lib\corvus.lib") {
    $capCorvus = @("$capPrefix\lib\corvus.lib", "$capPrefix\lib\hwy.lib", "$capPrefix\lib\hwy_contrib.lib")
    $capInc    = @("$new\include", "$new\include\libstats", "$new\build-bench-msvc-cap\generated")
    $capLib    = @("$new\build-bench-msvc-cap\stats_static.lib") + $capCorvus
    Build 'dist_v250_cap'      'distributions_bench.cpp'  $capInc $capLib
    Build 'elem_v250_cap'      'elementary_bench.cpp'     $capInc $capLib
    Build 'corvus_scaling_cap' 'corvus_scaling_bench.cpp' @("$capPrefix\include") $capCorvus
}

Get-ChildItem $out -Filter *.exe | Select-Object Name, Length | Format-Table -AutoSize
