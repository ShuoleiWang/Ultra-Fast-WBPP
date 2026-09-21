#Requires -Version 5.1
<#
.SYNOPSIS
    Install the Ultra-Fast WBPP MSI silently, attest the installed worker runtime, uninstall.

.DESCRIPTION
    Release gate for the Windows installer. The attestation must run against the
    tree that msiexec actually lays down, not against the build directory, so
    that the resource path the desktop app resolves at run time, the WiX file
    table and the frozen worker are all proven together:

      1. msiexec /i <msi> /qn /norestart /L*v <log>
         The Tauri MSI is per-machine (Program Files), so this needs an
         elevated shell; GitHub's Windows runners provide one.
      2. Locate the installation through the Uninstall registry entry the MSI
         writes (ARPINSTALLLOCATION) instead of assuming Program Files.
      3. Check the layout the app's sidecar discovery expects
         (resource_dir() is the executable directory on Windows):
           <InstallLocation>\<main binary>.exe   (exactly one .exe at the root)
           <InstallLocation>\resources\openastroflow-worker\openastroflow-worker-<target>.manifest.json
           <InstallLocation>\resources\openastroflow-worker\openastroflow-worker-<target>\openastroflow-worker-<target>.exe
      4. python scripts\attest_bundled_runtime.py --resource-root ... --target ... --output ...
         (PE import closure against Windows system DLLs, static-CRT kernel DLL,
         launch budget, doctor proving the kernels load from the installed tree).
      5. msiexec /x <msi> /qn /norestart, then require the install directory to
         be gone: files left behind by the uninstaller are a product defect.

    The uninstall runs even when the attestation fails (unless -KeepInstalled),
    and the script exits non-zero on any failure. Authenticode signing is a
    separate, documented release gate and is not checked here.

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File scripts\windows\attest-installed-msi.ps1 `
        -Msi "target\release\bundle\msi\Ultra-Fast WBPP_0.1.0_x64_en-US.msi" `
        -Target x86_64-pc-windows-msvc `
        -Output build\bundle-attestations\openastroflow-worker-x86_64-pc-windows-msvc.bundled.manifest.json
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)] [string]$Msi,
    [Parameter(Mandatory)] [string]$Target,
    [Parameter(Mandatory)] [string]$Output,
    [string]$Python = "python",
    [double]$MaxStartSeconds = 15,
    [switch]$KeepInstalled
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

function Write-Step([string]$Message) {
    Write-Host ("[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $Message)
}

function Invoke-Msiexec([string[]]$Arguments) {
    Write-Step ("msiexec " + ($Arguments -join " "))
    $process = Start-Process -FilePath "msiexec.exe" -ArgumentList $Arguments -Wait -PassThru -NoNewWindow
    # 0 = success, 3010 = success but a reboot is pending (files were in use).
    if ($process.ExitCode -ne 0 -and $process.ExitCode -ne 3010) {
        throw "msiexec exited with $($process.ExitCode)"
    }
    return $process.ExitCode
}

function Get-EntryProperty($Entry, [string]$Name) {
    # Registry entries lack most properties; under Set-StrictMode Windows
    # PowerShell 5.1 refuses both `$entry.Missing` and member enumeration of
    # `.PSObject.Properties.Name`, so enumerate the property objects instead.
    if ($null -eq $Entry) { return $null }
    $property = @($Entry.PSObject.Properties | Where-Object { $_.Name -eq $Name })
    if ($property.Count -eq 0) { return $null }
    return $property[0].Value
}

function Get-InstalledProduct([string]$DisplayName) {
    $roots = @(
        "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall",
        "HKLM:\SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall"
    )
    # Not "$matches": that name is PowerShell's automatic regex-match variable.
    $found = @()
    foreach ($root in $roots) {
        if (-not (Test-Path $root)) { continue }
        foreach ($key in Get-ChildItem $root) {
            # A key without values yields nothing here (not an empty object).
            $entry = Get-ItemProperty -Path $key.PSPath -ErrorAction SilentlyContinue
            if ($null -eq $entry) { continue }
            if ([string](Get-EntryProperty $entry "DisplayName") -eq $DisplayName) {
                $found += $entry
            }
        }
    }
    return $found
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$msiPath = (Resolve-Path $Msi).Path
if (-not (Test-Path $msiPath -PathType Leaf)) { throw "MSI not found: $msiPath" }
if (Test-Path $Output) { throw "attestation output must not exist yet: $Output" }
$tauriConfig = Get-Content (Join-Path $repoRoot "apps\desktop\src-tauri\tauri.conf.json") -Raw | ConvertFrom-Json
$productName = [string]$tauriConfig.productName
if (-not $productName) { throw "tauri.conf.json has no productName" }

if (Get-InstalledProduct $productName) {
    throw "$productName is already installed on this machine; uninstall it before running the gate"
}

$log = Join-Path ([System.IO.Path]::GetTempPath()) ("ultra-fast-wbpp-msi-{0}.log" -f (Get-Date -Format "yyyyMMdd-HHmmss"))
$installCode = Invoke-Msiexec @("/i", "`"$msiPath`"", "/qn", "/norestart", "/L*v", "`"$log`"")
Write-Step "install exit code $installCode (log: $log)"

# Failures are collected rather than thrown immediately so that the uninstall
# always runs and an attestation failure is never masked by a cleanup failure.
$failures = @()
$installLocation = $null
try {
    $products = @(Get-InstalledProduct $productName)
    if ($products.Count -ne 1) {
        throw "expected exactly one Uninstall entry for '$productName', found $($products.Count)"
    }
    $installLocation = [string](Get-EntryProperty $products[0] "InstallLocation")
    if (-not $installLocation) { throw "the Uninstall entry has no InstallLocation (ARPINSTALLLOCATION)" }
    $installLocation = $installLocation.Trim('"').TrimEnd('\')
    Write-Step "installed at $installLocation"

    # The layout the desktop app resolves: resource_dir() is the exe directory.
    # Tauri keeps the Cargo binary name (openastroflow-desktop.exe) unless
    # mainBinaryName is configured, so the executable is located, not assumed:
    # the install root must hold exactly one .exe.
    $rootExecutables = @(Get-ChildItem -Path $installLocation -File -Filter "*.exe")
    if ($rootExecutables.Count -ne 1) {
        throw ("expected exactly one executable in {0}, found {1}" -f $installLocation, (($rootExecutables | ForEach-Object { $_.Name }) -join ", "))
    }
    $mainExe = $rootExecutables[0].FullName
    $resourceRoot = Join-Path $installLocation "resources\openastroflow-worker"
    $manifest = Join-Path $resourceRoot "openastroflow-worker-$Target.manifest.json"
    $workerExe = Join-Path $resourceRoot "openastroflow-worker-$Target\openastroflow-worker-$Target.exe"
    foreach ($required in @($mainExe, $manifest, $workerExe)) {
        if (-not (Test-Path $required -PathType Leaf)) {
            throw "installed layout is missing $required"
        }
    }
    Write-Step "sidecar layout present under $resourceRoot"

    $outputDir = Split-Path -Parent $Output
    if ($outputDir -and -not (Test-Path $outputDir)) { New-Item -ItemType Directory -Path $outputDir | Out-Null }
    $attest = @(
        (Join-Path $repoRoot "scripts\attest_bundled_runtime.py"),
        "--resource-root", $resourceRoot,
        "--target", $Target,
        "--max-start-seconds", $MaxStartSeconds,
        "--main-executable", $mainExe,
        "--output", $Output
    )
    Write-Step ("$Python " + ($attest -join " "))
    Push-Location $repoRoot
    try {
        & $Python @attest
        if ($LASTEXITCODE -ne 0) { throw "attest_bundled_runtime.py exited with $LASTEXITCODE" }
    } finally {
        Pop-Location
    }
    Write-Step "attestation written to $Output"
} catch {
    $failures += "installed-runtime attestation failed: $($_.Exception.Message)"
}

if ($KeepInstalled) {
    Write-Step "-KeepInstalled: leaving $productName installed"
} else {
    try {
        $uninstallCode = Invoke-Msiexec @("/x", "`"$msiPath`"", "/qn", "/norestart")
        Write-Step "uninstall exit code $uninstallCode"
        if ($installLocation -and (Test-Path $installLocation)) {
            $leftovers = @(Get-ChildItem -Path $installLocation -Recurse -Force -File | Select-Object -First 20 | ForEach-Object { $_.FullName })
            throw ("uninstall left files behind under {0}: {1}" -f $installLocation, ($leftovers -join "; "))
        }
        if (Get-InstalledProduct $productName) {
            throw "uninstall left an Uninstall registry entry for $productName"
        }
        Write-Step "uninstall removed the installation directory and registry entry"
    } catch {
        $failures += "uninstall check failed: $($_.Exception.Message)"
    }
}

if ($failures.Count -gt 0) {
    throw ($failures -join " | ")
}
