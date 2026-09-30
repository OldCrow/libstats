# Build Highway 1.4.0 + corvus v1.0.1 and install both to a prefix that an MSVC-built libstats
# consumes through find_package(corvus).
#   -Compiler clang-cl : MSVC ABI, corvus reaches AVX3_ZEN4   -> build-bench-deps\prefix
#   -Compiler cl       : corvus stops at the MSVC AVX2 cap     -> build-bench-deps\prefix-msvc
param(
    [Parameter(Mandatory)] [ValidateSet('clang-cl', 'cl')] [string] $Compiler
)
$ErrorActionPreference = 'Stop'

$vcvars = 'C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat'
cmd /c "`"$vcvars`" > nul && set" | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') { Set-Item -Path "env:$($Matches[1])" -Value $Matches[2] }
}
if ($Compiler -eq 'clang-cl') {
    $env:PATH = 'C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\Llvm\x64\bin;' + $env:PATH
}

$sfx    = if ($Compiler -eq 'cl') { '-msvc' } else { '' }
$root   = 'C:\Users\gdwol\Development\libstats\build-bench-deps'
$prefix = "$root\prefix$sfx"
$hwySrc = 'C:\Users\gdwol\Development\corvus\build-clangcl\_deps\highway-src'
$corSrc = 'C:\Users\gdwol\Development\corvus'

cmake -S $hwySrc -B "$root\highway$sfx" -G Ninja -DCMAKE_BUILD_TYPE=Release `
    "-DCMAKE_C_COMPILER=$Compiler" "-DCMAKE_CXX_COMPILER=$Compiler" `
    -DHWY_ENABLE_TESTS=OFF -DHWY_ENABLE_EXAMPLES=OFF -DHWY_ENABLE_CONTRIB=ON `
    -DHWY_FORCE_STATIC_LIBS=ON -DBUILD_TESTING=OFF "-DCMAKE_INSTALL_PREFIX=$prefix"
if ($LASTEXITCODE -ne 0) { throw 'highway configure failed' }
cmake --build "$root\highway$sfx" --parallel
if ($LASTEXITCODE -ne 0) { throw 'highway build failed' }
cmake --install "$root\highway$sfx" | Select-Object -Last 3
if ($LASTEXITCODE -ne 0) { throw 'highway install failed' }

cmake -S $corSrc -B "$root\corvus$sfx" -G Ninja -DCMAKE_BUILD_TYPE=Release `
    "-DCMAKE_C_COMPILER=$Compiler" "-DCMAKE_CXX_COMPILER=$Compiler" `
    -DCORVUS_BUILD_TESTS=OFF -DCORVUS_BUILD_EXAMPLES=OFF `
    "-DCMAKE_PREFIX_PATH=$prefix" "-DCMAKE_INSTALL_PREFIX=$prefix"
if ($LASTEXITCODE -ne 0) { throw 'corvus configure failed' }
cmake --build "$root\corvus$sfx" --parallel
if ($LASTEXITCODE -ne 0) { throw 'corvus build failed' }
cmake --install "$root\corvus$sfx" | Select-Object -Last 3
if ($LASTEXITCODE -ne 0) { throw 'corvus install failed' }

Get-ChildItem "$prefix\lib" -Filter *.lib | Select-Object Name, Length | Format-Table -AutoSize
