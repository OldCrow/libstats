# Item 3: corvus v1.0.1 built with clang-cl but capped at AVX2, to separate compiler from tier in
# the 5-30x gap between the clang-cl (AVX3_ZEN4) and MSVC (AVX2) builds. Then corvus_scaling
# against it, compiled with cl and the same flags as the other bench binaries.
$ErrorActionPreference = 'Stop'
$vcvars = 'C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat'
cmd /c "`"$vcvars`" > nul && set" | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') { Set-Item -Path "env:$($Matches[1])" -Value $Matches[2] }
}
$llvm   = 'C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\Llvm\x64\bin'
$new    = 'C:\Users\gdwol\Development\libstats'
$prefix = "$new\build-bench-deps\prefix"                 # clang-cl Highway 1.4.0 (all targets)
$corSrc = 'C:\Users\gdwol\Development\corvus-v1.0.1'     # worktree at tag v1.0.1
$corBin = "$new\build-bench-deps\corvus-clangcl-avx2"
$cap    = 'HWY_AVX10_2|HWY_AVX3_SPR|HWY_AVX3_ZEN4|HWY_AVX3_DL|HWY_AVX3'

$savedPath = $env:PATH
$env:PATH = "$llvm;" + $env:PATH
cmake -S $corSrc -B $corBin -G Ninja -DCMAKE_BUILD_TYPE=Release `
    '-DCMAKE_C_COMPILER=clang-cl' '-DCMAKE_CXX_COMPILER=clang-cl' `
    -DCORVUS_BUILD_TESTS=OFF -DCORVUS_BUILD_EXAMPLES=OFF `
    "-DCORVUS_DISABLED_TARGETS=$cap" "-DCMAKE_PREFIX_PATH=$prefix"
if ($LASTEXITCODE -ne 0) { throw 'corvus configure failed' }
cmake --build $corBin --parallel
if ($LASTEXITCODE -ne 0) { throw 'corvus build failed' }
$env:PATH = $savedPath

$out = "$new\build-bench-msvc\bench"
Set-Location $out
$cxx = @('/nologo', '/std:c++20', '/O2', '/Ob2', '/DNDEBUG', '/EHsc', '/MD', '/utf-8', '/arch:AVX512',
         '/DNOMINMAX', '/D_USE_MATH_DEFINES', '/D_CRT_SECURE_NO_WARNINGS', '/D_CRT_NONSTDC_NO_DEPRECATE')
& cl @cxx "/I$corSrc\include" "$new\tools\bench\corvus_scaling_bench.cpp" `
    '/Fe:corvus_scaling_clangcl_avx2.exe' '/Fo:corvus_scaling_clangcl_avx2.obj' `
    /link "$corBin\corvus.lib" "$prefix\lib\hwy.lib" "$prefix\lib\hwy_contrib.lib"
if ($LASTEXITCODE -ne 0) { throw 'bench build failed' }
Get-ChildItem $out -Filter 'corvus_scaling_clangcl_avx2.exe' | Select-Object Name, Length | Format-Table -AutoSize
