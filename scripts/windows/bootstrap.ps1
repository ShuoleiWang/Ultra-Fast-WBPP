#Requires -Version 5.1
<#
.SYNOPSIS
    Install the Windows x86-64 developer toolchain for Ultra-Fast WBPP.

.DESCRIPTION
    Idempotent winget bootstrap for a Windows 10/11 x86-64 build or test machine:

      * Python 3.12            (Python.Python.3.12)         engine + tests
      * CMake                  (Kitware.CMake)              native kernels
      * Ninja                  (Ninja-build.Ninja)          native kernels
      * VS 2022 Build Tools    (Microsoft.VisualStudio.2022.BuildTools)
                               with the "Desktop development with C++" workload
                               (MSVC v143, Windows SDK) -- the only supported
                               compiler for native on Windows
      * uv                     (astral-sh.uv)               fast venv/pip
      * rustup                 (Rustlang.Rustup)            only with -WithRust

    Every package that is already present is skipped; every action and every
    resulting tool version is written to a JSON report (default:
    bootstrap-report.json next to this script) so the machine's toolchain can be
    quoted in receipts and reports. Nothing else on the machine is changed: no
    system settings, no services, no scheduled tasks.

    Run from an elevated PowerShell (Build Tools and CMake install machine-wide):

        powershell -NoProfile -ExecutionPolicy Bypass -File scripts\windows\bootstrap.ps1

    Package installs do not update the PATH of the calling shell; open a new
    shell afterwards (or call scripts\windows\dev-env.ps1) before building.
#>
[CmdletBinding()]
param(
    [string]$ReportPath = "",
    [switch]$WithRust,
    [switch]$SkipBuildTools
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

if (-not $ReportPath) {
    $ReportPath = Join-Path $PSScriptRoot "bootstrap-report.json"
}

function Write-Step([string]$Message) {
    Write-Host ("[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $Message)
}

function Refresh-Path {
    $machine = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $user = [Environment]::GetEnvironmentVariable("Path", "User")
    $env:PATH = "$machine;$user"
}

function Test-WingetPackage([string]$Id) {
    & winget list --id $Id --exact --accept-source-agreements --disable-interactivity *> $null
    return ($LASTEXITCODE -eq 0)
}

function Install-WingetPackage {
    param(
        [Parameter(Mandatory)] [string]$Id,
        [string[]]$ExtraArguments = @()
    )
    $arguments = @(
        "install", "--id", $Id, "--exact", "--silent",
        "--accept-package-agreements", "--accept-source-agreements",
        "--disable-interactivity"
    ) + $ExtraArguments
    Write-Step ("winget " + ($arguments -join " "))
    & winget @arguments
    $code = $LASTEXITCODE
    # 0 = installed; -1978335189 (0x8A15002B) = no applicable upgrade / already installed.
    if ($code -ne 0 -and $code -ne -1978335189) {
        throw "winget install $Id failed with exit code $code"
    }
}

function Get-VsBuildToolsPath {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) { return $null }
    $path = & $vswhere -products Microsoft.VisualStudio.Product.BuildTools `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
        -property installationPath -latest 2>$null
    if ($LASTEXITCODE -ne 0 -or -not $path) { return $null }
    return $path.Trim()
}

function Get-ToolVersion([string]$Command, [string[]]$Arguments) {
    try {
        $output = & $Command @Arguments 2>&1 | Select-Object -First 1
        return [string]$output
    } catch {
        return $null
    }
}

$started = Get-Date
$actions = New-Object System.Collections.ArrayList
$packages = @(
    @{ Id = "Python.Python.3.12"; Name = "Python 3.12"; Arguments = @("--scope", "user") },
    @{ Id = "Kitware.CMake"; Name = "CMake"; Arguments = @() },
    @{ Id = "Ninja-build.Ninja"; Name = "Ninja"; Arguments = @() },
    @{ Id = "astral-sh.uv"; Name = "uv"; Arguments = @() }
)
if ($WithRust) {
    $packages += @{ Id = "Rustlang.Rustup"; Name = "rustup"; Arguments = @() }
}

Write-Step ("winget " + (& winget --version))
foreach ($package in $packages) {
    if (Test-WingetPackage $package.Id) {
        Write-Step ("{0} already installed; skipping" -f $package.Name)
        [void]$actions.Add(@{ id = $package.Id; name = $package.Name; action = "already-installed" })
        continue
    }
    Install-WingetPackage -Id $package.Id -ExtraArguments $package.Arguments
    [void]$actions.Add(@{ id = $package.Id; name = $package.Name; action = "installed" })
}

if (-not $SkipBuildTools) {
    $buildTools = Get-VsBuildToolsPath
    if ($buildTools) {
        Write-Step "VS 2022 Build Tools with MSVC already installed at $buildTools; skipping"
        [void]$actions.Add(@{ id = "Microsoft.VisualStudio.2022.BuildTools"; name = "VS 2022 Build Tools"; action = "already-installed" })
    } else {
        $override = "--quiet --wait --norestart --nocache --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
        Install-WingetPackage -Id "Microsoft.VisualStudio.2022.BuildTools" -ExtraArguments @("--override", $override)
        [void]$actions.Add(@{ id = "Microsoft.VisualStudio.2022.BuildTools"; name = "VS 2022 Build Tools"; action = "installed"; override = $override })
    }
}

Refresh-Path

$pythonExe = $null
foreach ($candidate in @(
        (Join-Path $env:LOCALAPPDATA "Programs\Python\Python312\python.exe"),
        (Join-Path $env:ProgramFiles "Python312\python.exe"))) {
    if (Test-Path $candidate) { $pythonExe = $candidate; break }
}
$buildToolsPath = Get-VsBuildToolsPath
$msvcVersion = $null
if ($buildToolsPath) {
    $versionFile = Join-Path $buildToolsPath "VC\Auxiliary\Build\Microsoft.VCToolsVersion.default.txt"
    if (Test-Path $versionFile) { $msvcVersion = (Get-Content $versionFile -First 1).Trim() }
}

$report = [ordered]@{
    schemaVersion = 1
    startedAt = $started.ToString("o")
    finishedAt = (Get-Date).ToString("o")
    machine = [ordered]@{
        computerName = $env:COMPUTERNAME
        windows = (Get-CimInstance Win32_OperatingSystem).Caption
        build = (Get-CimInstance Win32_OperatingSystem).BuildNumber
        cpu = (Get-CimInstance Win32_Processor | Select-Object -First 1).Name
        logicalProcessors = (Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors
        physicalMemoryBytes = (Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory
    }
    actions = $actions.ToArray()
    tools = [ordered]@{
        winget = (& winget --version)
        python = if ($pythonExe) { Get-ToolVersion $pythonExe @("--version") } else { $null }
        pythonPath = $pythonExe
        cmake = Get-ToolVersion "cmake" @("--version")
        ninja = Get-ToolVersion "ninja" @("--version")
        uv = Get-ToolVersion "uv" @("--version")
        git = Get-ToolVersion "git" @("--version")
        rustc = if ($WithRust) { Get-ToolVersion "rustc" @("--version") } else { $null }
        vsBuildToolsPath = $buildToolsPath
        msvcToolsVersion = $msvcVersion
    }
}
$report | ConvertTo-Json -Depth 5 | Set-Content -Path $ReportPath -Encoding UTF8
Write-Step "report written to $ReportPath"
Write-Step ("done in {0:n0} s" -f ((Get-Date) - $started).TotalSeconds)
