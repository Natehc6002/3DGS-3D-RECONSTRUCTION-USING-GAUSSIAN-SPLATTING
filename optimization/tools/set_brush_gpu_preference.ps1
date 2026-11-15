# Set (or undo) the Windows per-app GPU preference for Brush only.
#
# This is the same per-user registry value written by
#   Settings > System > Display > Graphics > High performance
# for an app: HKCU:\Software\Microsoft\DirectX\UserGpuPreferences
# value name = full path to the exe, data = "GpuPreference=2;".
# HKCU only (current user), no admin, no HKLM.
#
# Usage (run from an elevated-free PowerShell as your normal user):
#   .\set_brush_gpu_preference.ps1                  # show current value
#   .\set_brush_gpu_preference.ps1 -Apply           # set High performance
#   .\set_brush_gpu_preference.ps1 -Undo            # remove only this value
#
# AUTHORIZED PATH: exactly C:\Tools\Brush\brush_app.exe and nothing else.
# Any other path is refused. This script never touches other values.
#
# After -Apply you MUST fully close Brush (all windows) and reopen it;
# Windows reads this preference when the process starts.
param(
    [switch]$Apply,
    [switch]$Undo,

    [string]$BrushExe = "C:\Tools\Brush\brush_app.exe"
)

$ErrorActionPreference = "Stop"
$allowed = "C:\Tools\Brush\brush_app.exe"

function Fail([string]$Message) {
    [Console]::Error.WriteLine("ERROR: $Message")
    exit 2
}

if ($Apply -and $Undo) {
    Fail "Use either -Apply or -Undo, not both."
}

# Refuse any path other than the single authorized one (normalise
# separators and case before comparing).
$norm = $BrushExe -replace '/', '\'
$allowedNorm = $allowed -replace '/', '\'
if (-not [string]::Equals($norm, $allowedNorm,
        [System.StringComparison]::OrdinalIgnoreCase)) {
    Fail ("Refusing: this script only operates on '$allowed'. " +
          "Got '$BrushExe'.")
}

$key = "HKCU:\Software\Microsoft\DirectX\UserGpuPreferences"
$name = $allowed
$desired = "GpuPreference=2;"

$existing = $null
if (Test-Path -LiteralPath $key) {
    $existing = (Get-Item -LiteralPath $key).GetValue($name, $null)
}

if (-not $Apply -and -not $Undo) {
    if ($null -eq $existing) {
        Write-Host "Current value for $name : (not set)"
    } else {
        Write-Host "Current value for $name : $existing"
    }
    Write-Host "Run with -Apply to set '$desired' (High performance),"
    Write-Host "or -Undo to remove it."
    exit 0
}

if ($Undo) {
    if ($null -eq $existing) {
        Write-Host "Nothing to undo: $name is not set."
        exit 0
    }
    Write-Host "Previous value: $existing"
    Remove-ItemProperty -LiteralPath $key -Name $name
    Write-Host "Removed value: $name"
    Write-Host "Fully close Brush and reopen it for the change to apply."
    exit 0
}

# -Apply
Write-Host "Previous value: $(if ($null -eq $existing) { '(not set)' } else { $existing })"
if (-not (Test-Path -LiteralPath $key)) {
    New-Item -Path $key -Force | Out-Null
}
New-ItemProperty -LiteralPath $key -Name $name -Value $desired `
    -PropertyType String -Force | Out-Null
$new = (Get-Item -LiteralPath $key).GetValue($name, $null)
Write-Host "New value: $new"
Write-Host "Set HKCU per-app GPU preference for $name"
Write-Host "IMPORTANT: fully close Brush (all windows) and reopen it now."
exit 0
