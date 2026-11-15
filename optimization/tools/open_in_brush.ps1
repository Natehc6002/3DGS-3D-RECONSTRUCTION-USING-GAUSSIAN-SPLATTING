# Open a PLY in Brush 0.3.0 with NVIDIA-preference environment variables.
#
# Usage:
#   .\open_in_brush.ps1 -Ply <path\to\file.ply>
#   .\open_in_brush.ps1 -Ply <path> -BrushExe C:\Tools\Brush\brush_app.exe
#
# The three WGPU_* variables below are set for the CHILD process only
# (they do not change your user or system environment).
#
# IMPORTANT: Brush 0.3.0 is NOT guaranteed to honour these variables.
# WGPU_ADAPTER_NAME in particular is a wgpu-native hint that the app may
# ignore; our screen recording showed Brush still running on the Intel
# iGPU with WGPU_BACKEND/WGPU_POWER_PREF set. Verify the actual adapter
# in Brush's Stats panel (must show an NVIDIA name and DiscreteGpu) -
# see README_brush_gpu.md. Do not assume this script fixed anything
# until the Stats panel confirms it.
param(
    [Parameter(Mandatory = $true)]
    [string]$Ply,

    [string]$BrushExe = "C:\Tools\Brush\brush_app.exe"
)

function Fail([string]$Message) {
    [Console]::Error.WriteLine("ERROR: $Message")
    exit 2
}

if (-not (Test-Path -LiteralPath $Ply -PathType Leaf)) {
    Fail "PLY not found: $Ply"
}
if ([System.IO.Path]::GetExtension($Ply) -ne ".ply") {
    Fail "Not a .ply file: $Ply"
}
if (-not (Test-Path -LiteralPath $BrushExe -PathType Leaf)) {
    Fail "Brush executable not found: $BrushExe"
}

$plyFull = (Resolve-Path -LiteralPath $Ply).Path
$brushFull = (Resolve-Path -LiteralPath $BrushExe).Path

# Child-process-only environment (see header comment about guarantees).
$env:WGPU_BACKEND = "vulkan"
$env:WGPU_POWER_PREF = "high"
$env:WGPU_ADAPTER_NAME = "NVIDIA"

$cmd = "`"$brushFull`" --with-viewer `"$plyFull`""
Write-Host "Command: $cmd"
Write-Host "Env (child only): WGPU_BACKEND=vulkan WGPU_POWER_PREF=high WGPU_ADAPTER_NAME=NVIDIA"
Write-Host "Reminder: check Brush's Stats panel - GPU must be NVIDIA / DiscreteGpu."

& $brushFull --with-viewer $plyFull
exit $LASTEXITCODE
