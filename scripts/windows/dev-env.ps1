#Requires -Version 5.1
<#
.SYNOPSIS
    Prepare the current PowerShell session for building and testing Ultra-Fast WBPP on Windows x86-64.

.DESCRIPTION
    Dot-source this script after scripts\windows\bootstrap.ps1 (or on any machine
    with the toolchain already installed) instead of opening a new shell:

        . scripts\windows\dev-env.ps1

    It changes only this session:

      * reloads PATH from the registry so tools installed by winget are visible
        without a new console;
      * imports the MSVC x64 environment of the newest VS 2022 Build Tools
        (VsDevCmd.bat -arch=amd64), which CMake's Ninja generator and the
        /W4 /WX /fp:strict native build need;
      * activates .venv when the repository has one (skip with -NoVenv);
      * sets PYTHONUTF8=1 so the console never re-encodes the engine's UTF-8
        receipts and progress lines through the OEM code page;
      * prints the tool versions it found and the long-path policy, which the
        engine's path budget reports as well (LongPathsEnabled under
        HKLM\SYSTEM\CurrentControlSet\Control\FileSystem).

    Nothing is installed or written anywhere; run bootstrap.ps1 for that.
#>
[CmdletBinding()]
param(
    [switch]$NoVenv,
    [switch]$NoMsvc
)

$ErrorActionPreference = "Stop"

function Write-DevEnvStep([string]$Message) {
    Write-Host ("[dev-env] {0}" -f $Message)
}

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")

# 1. PATH as a freshly opened console would see it, plus whatever this
#    session already added (deduplicated, order preserved).
$machinePath = [Environment]::GetEnvironmentVariable("Path", "Machine")
$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
$merged = New-Object System.Collections.Generic.List[string]
foreach ($entry in (($machinePath, $userPath, $env:PATH) -join ";").Split(";")) {
    $trimmed = $entry.Trim()
    if ($trimmed -and -not $merged.Contains($trimmed)) { [void]$merged.Add($trimmed) }
}
$env:PATH = ($merged -join ";")

# 2. MSVC x64 environment from the newest Build Tools with the C++ workload.
if (-not $NoMsvc) {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    $installation = $null
    if (Test-Path $vswhere) {
        $installation = & $vswhere -latest -products * `
            -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
            -property installationPath 2>$null
        if ($LASTEXITCODE -ne 0) { $installation = $null }
    }
    $vsDevCmd = if ($installation) { Join-Path $installation.Trim() "Common7\Tools\VsDevCmd.bat" } else { $null }
    if ($vsDevCmd -and (Test-Path $vsDevCmd)) {
        # Run the batch file in cmd.exe and copy the variables it exports;
        # VsDevCmd cannot be sourced by PowerShell directly.
        $exported = & cmd.exe /d /s /c "`"$vsDevCmd`" -arch=amd64 -host_arch=amd64 -no_logo && set"
        foreach ($line in $exported) {
            $separator = $line.IndexOf("=")
            if ($separator -gt 0) {
                $name = $line.Substring(0, $separator)
                $value = $line.Substring($separator + 1)
                Set-Item -Path ("Env:" + $name) -Value $value
            }
        }
        Write-DevEnvStep ("MSVC x64 environment imported from {0}" -f $installation.Trim())
    } else {
        Write-Warning "VS 2022 Build Tools with the C++ workload were not found; the native build needs them (scripts\windows\bootstrap.ps1)."
    }
}

# 3. The repository's virtual environment, when it exists.
$activate = Join-Path $repoRoot ".venv\Scripts\Activate.ps1"
if (-not $NoVenv -and (Test-Path $activate)) {
    . $activate
    Write-DevEnvStep "activated .venv"
} elseif (-not $NoVenv) {
    Write-DevEnvStep ".venv not found; create it with: python -m venv .venv"
}

# 4. UTF-8 everywhere the engine talks to the console.
$env:PYTHONUTF8 = "1"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# 5. Summary.
function Get-DevEnvToolVersion([string]$Command, [string[]]$Arguments) {
    $resolved = Get-Command $Command -ErrorAction SilentlyContinue
    if (-not $resolved) { return "not found" }
    try {
        $output = & $resolved.Source @Arguments 2>&1 | Select-Object -First 1
        return [string]$output
    } catch {
        return "present ($($resolved.Source))"
    }
}

$longPaths = $null
try {
    $longPaths = (Get-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name LongPathsEnabled -ErrorAction Stop).LongPathsEnabled
} catch {
    $longPaths = 0
}

Write-DevEnvStep ("python : {0}" -f (Get-DevEnvToolVersion "python" @("--version")))
Write-DevEnvStep ("cmake  : {0}" -f (Get-DevEnvToolVersion "cmake" @("--version")))
Write-DevEnvStep ("ninja  : {0}" -f (Get-DevEnvToolVersion "ninja" @("--version")))
Write-DevEnvStep ("cl     : {0}" -f (Get-DevEnvToolVersion "cl" @()))
Write-DevEnvStep ("uv     : {0}" -f (Get-DevEnvToolVersion "uv" @("--version")))
Write-DevEnvStep ("git    : {0}" -f (Get-DevEnvToolVersion "git" @("--version")))
Write-DevEnvStep ("rustc  : {0}" -f (Get-DevEnvToolVersion "rustc" @("--version")))
if ($longPaths -eq 1) {
    Write-DevEnvStep "long paths: enabled (LongPathsEnabled=1); output paths may exceed 259 characters"
} else {
    Write-DevEnvStep "long paths: disabled (MAX_PATH, 259 characters); keep output directories short or enable the policy as an administrator"
}
Write-DevEnvStep ("repository: {0}" -f $repoRoot)
