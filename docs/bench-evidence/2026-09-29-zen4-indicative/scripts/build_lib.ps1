# Configure + build libstats_static (Release, Ninja) for the tools/bench comparatives.
# Usage: build_lib.ps1 -Src <checkout> -Compiler clang-cl|cl
param(
    [Parameter(Mandatory)] [string] $Src,
    [Parameter(Mandatory)] [ValidateSet('clang-cl', 'cl')] [string] $Compiler,
    [string] $Prefix = '',  # install prefix holding a system corvus + Highway (NEW side only)
    [string] $Tag = ''      # build dir suffix; default msvc / clangcl by compiler
)
$ErrorActionPreference = 'Stop'

$vcvars = 'C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat'
cmd /c "`"$vcvars`" > nul && set" | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') { Set-Item -Path "env:$($Matches[1])" -Value $Matches[2] }
}
if ($Compiler -eq 'clang-cl') {
    # Same clang-cl the corvus windows-clang-cl tree was validated with (VS-bundled).
    $env:PATH = 'C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\Llvm\x64\bin;' + $env:PATH
}

if (-not $Tag) { $Tag = if ($Compiler -eq 'cl') { 'msvc' } else { 'clangcl' } }
$bin = Join-Path $Src "build-bench-$Tag"

$extra = @()
if ($Prefix) { $extra += "-DCMAKE_PREFIX_PATH=$Prefix" }

cmake -S $Src -B $bin -G Ninja `
    -DCMAKE_BUILD_TYPE=Release `
    "-DCMAKE_C_COMPILER=$Compiler" "-DCMAKE_CXX_COMPILER=$Compiler" `
    -DLIBSTATS_BUILD_TESTS=OFF -DLIBSTATS_BUILD_TOOLS=OFF -DLIBSTATS_BUILD_EXAMPLES=OFF @extra
if ($LASTEXITCODE -ne 0) { throw "configure failed ($LASTEXITCODE)" }

cmake --build $bin --target libstats_static --parallel
if ($LASTEXITCODE -ne 0) { throw "build failed ($LASTEXITCODE)" }

Get-ChildItem $bin -Recurse -Include *.lib | Select-Object FullName, Length | Format-Table -AutoSize
