# Open a PLY in Brush with an explicit GPU selection (child process only).
#
# Usage:
#   .\open_in_brush.ps1 -Ply <path\to\file.ply> [-Gpu auto|nvidia|intel] [-PrintOnly]
#   .\open_in_brush.ps1 -Ply <path> -BrushExe C:\Tools\Brush\brush_app.exe
#
# GPU modes (evidence and caveats: README_brush_gpu.md):
#   nvidia  point VK_DRIVER_FILES at the newest NVIDIA Vulkan ICD json
#           (nv-vk64.json) for the Brush child process; a clean error if
#           the json cannot be found.
#   auto    same as nvidia when the json exists, otherwise warn clearly
#           and fall back to the default adapter (no VK_DRIVER_FILES).
#   intel   never touch VK_DRIVER_FILES: default adapter (the Intel iGPU
#           on this machine).
#
# In every mode WGPU_BACKEND=vulkan is set for the child, because the
# DirectX 12 backend crashes Brush 0.3.0 on this machine (FXC X3695).
# VK_DRIVER_FILES and WGPU_BACKEND are set for the CHILD only: the script
# saves your session values and restores them before it exits.
# A placeholder or wrong VK_DRIVER_FILES path makes Brush fail with
# "WGPU error: Failed to create surface for any enabled backend", so the
# path is validated before launch. Expected failures (missing PLY, missing
# Brush, missing json) print one ERROR line and exit 2 - no stack traces.
param(
    [Parameter(Mandatory = $true)]
    [string]$Ply,

    [ValidateSet("auto", "nvidia", "intel")]
    [string]$Gpu = "auto",

    [switch]$PrintOnly,

    [string]$BrushExe = "C:\Tools\Brush\brush_app.exe"
)

$ErrorActionPreference = "Stop"

$FileRepository = "C:\Windows\System32\DriverStore\FileRepository"

function Fail([string]$Message) {
    [Console]::Error.WriteLine("ERROR: $Message")
    exit 2
}

# Newest nv-vk64.json under the DriverStore (the folder name changes on
# every NVIDIA driver update, so we search by file name, not by path).
function Find-NvidiaVkIcd {
    if (-not (Test-Path -LiteralPath $FileRepository)) {
        return $null
    }
    $newest = Get-ChildItem -LiteralPath $FileRepository -Filter "nv-vk64.json" `
        -File -Recurse -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if ($null -eq $newest) {
        return $null
    }
    return $newest.FullName
}

try {
    # ---- expected failures: one clean line each, never a stack trace ----
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

    # ---- resolve the GPU mode ----
    $vkIcd = $null
    if ($Gpu -eq "intel") {
        $vkIcd = $null    # default adapter on purpose
    }
    else {
        $vkIcd = Find-NvidiaVkIcd
        if ($null -eq $vkIcd -and $Gpu -eq "nvidia") {
            Fail ("nv-vk64.json not found under $FileRepository " +
                  "(-Gpu nvidia requires it; use -Gpu auto to fall back " +
                  "to the default adapter)")
        }
        if ($null -eq $vkIcd) {
            Write-Warning ("nv-vk64.json not found under $FileRepository; " +
                "falling back to the default GPU adapter " +
                "(VK_DRIVER_FILES not set).")
        }
    }

    # ---- report ----
    $vkValue = if ($null -eq $vkIcd) {
        "(not set - default adapter)"
    }
    else {
        $vkIcd
    }
    $cmd = "`"$brushFull`" --with-viewer `"$plyFull`""
    Write-Host "GPU mode:        $Gpu"
    Write-Host "VK_DRIVER_FILES: $vkValue  (child only)"
    Write-Host "WGPU_BACKEND:    vulkan  (child only)"
    Write-Host "Command:         $cmd"
    if ($PrintOnly) {
        Write-Host "(dry run - Brush not launched)"
        exit 0
    }
    Write-Host "Reminder: check Brush's Stats panel after it opens."

    # ---- launch with child-only environment, then restore the session ----
    $prevVk = $env:VK_DRIVER_FILES
    $prevBackend = $env:WGPU_BACKEND
    try {
        if ($null -ne $vkIcd) {
            $env:VK_DRIVER_FILES = $vkIcd
        }
        else {
            Remove-Item -LiteralPath Env:VK_DRIVER_FILES -ErrorAction SilentlyContinue
        }
        $env:WGPU_BACKEND = "vulkan"
        & $brushFull --with-viewer $plyFull
        $code = $LASTEXITCODE
    }
    finally {
        if ($null -eq $prevVk) {
            Remove-Item -LiteralPath Env:VK_DRIVER_FILES -ErrorAction SilentlyContinue
        }
        else {
            $env:VK_DRIVER_FILES = $prevVk
        }
        if ($null -eq $prevBackend) {
            Remove-Item -LiteralPath Env:WGPU_BACKEND -ErrorAction SilentlyContinue
        }
        else {
            $env:WGPU_BACKEND = $prevBackend
        }
    }
    exit $code
}
catch {
    Fail $_.Exception.Message
}
