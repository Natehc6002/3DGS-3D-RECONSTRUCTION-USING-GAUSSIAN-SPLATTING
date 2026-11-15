# Brush GPU selection (make Brush use the NVIDIA RTX 4060)

Brush 0.3.0 on this machine has been observed running on
**Intel(R) UHD Graphics** (Stats panel: "Intel(R) RaptorLake-S Mobile
Graphics Controller", Type `IntegratedGpu`, driver 101.7026) even when
launched with `WGPU_BACKEND=vulkan` and `WGPU_POWER_PREF=high`. The real
GPU is an **NVIDIA GeForce RTX 4060 Laptop GPU** (driver 32.0.15.8186,
`Win32_VideoController` Status OK).

**This document does NOT claim any fix works.** Nothing here has been
verified in Brush 0.3.0 yet - the fix is only "confirmed" when the Steps
below show an NVIDIA name and `DiscreteGpu` in Brush's Stats panel.

## How to verify (do this after every change)

1. Open a PLY with the launcher:
   ```powershell
   .\optimization\tools\open_in_brush.ps1 -Ply <path\to\file.ply>
   ```
2. In Brush, open the **Stats** panel:
   - GPU Name must contain **NVIDIA** (currently it shows Intel).
   - Type must be **DiscreteGpu** (currently `IntegratedGpu`).
3. Open Task Manager -> Performance -> GPU: confirm the integrated GPU
   graph stays idle while Brush renders and the NVIDIA graph does the work.
4. If either check still shows Intel, the fix has **not** worked; nothing
   else in this project should be blamed for it.

Checklist for any file you open (record the result, "not tested" until
you look): Stats panel GPU name, colour shift vs original, flat shading,
holes, floaters, load errors.

## What the scripts do

### `open_in_brush.ps1` (child-process environment only)

```powershell
.\optimization\tools\open_in_brush.ps1 -Ply <path\to\file.ply>
# optional: -BrushExe C:\Tools\Brush\brush_app.exe (the default)
```

Sets `WGPU_BACKEND=vulkan`, `WGPU_POWER_PREF=high`,
`WGPU_ADAPTER_NAME=NVIDIA` for the Brush child process only, then prints
the exact command and launches Brush.

**Caveat:** these variables are a wgpu-native hint and are **not
guaranteed to be honoured by Brush 0.3.0**. In our recording, Brush still
picked the Intel iGPU with `WGPU_BACKEND`/`WGPU_POWER_PREF` set. Use the
Stats panel to check - never assume.

### `set_brush_gpu_preference.ps1` (per-user registry, not run by us yet)

Writes the same per-user value as
Settings > System > Display > Graphics > "High performance":
`HKCU\Software\Microsoft\DirectX\UserGpuPreferences`, value name =
full path to `brush_app.exe`, data = `GpuPreference=2;`.
Current user only (HKCU), no admin, no HKLM, and it refuses any path
other than `C:\Tools\Brush\brush_app.exe`.

**You run this yourself; we do not run it automatically:**

```powershell
# show current value (read-only)
.\optimization\tools\set_brush_gpu_preference.ps1
# set High performance (prints the previous value first)
.\optimization\tools\set_brush_gpu_preference.ps1 -Apply
# remove only this value
.\optimization\tools\set_brush_gpu_preference.ps1 -Undo
```

After `-Apply` you must **fully close Brush (all windows) and reopen
it**; Windows reads the preference when the process starts.

## Manual steps (equivalent, no scripts)

Windows: Settings -> System -> Display -> Graphics -> add
`C:\Tools\Brush\brush_app.exe` -> Options -> "High performance" -> Save.

NVIDIA Control Panel: Manage 3D settings -> Program Settings ->
Add `brush_app.exe` -> "High-performance NVIDIA processor" -> Apply.
(Exact wording varies by driver version.)

## If the Stats panel still shows Intel

1. NVIDIA Control Panel program setting for `brush_app.exe` (above).
2. Advanced Optimus / MUX switch: many laptops only expose the discrete
   GPU to apps when the dGPU drives the display - check your laptop
   vendor app (Lenovo Vantage, MyASUS, Armoury Crate, ...) or BIOS for
   "MUX switch" / "Discrete graphics" / "nVIDIA only" modes.
3. Update the NVIDIA driver (current on this machine: 32.0.15.8186) via
   NVIDIA GeForce Experience or the laptop vendor.

## Status

- Not verified for Brush 0.3.0 until confirmed in the Stats panel.
- Every Brush result in this project so far was obtained on the Intel
  integrated GPU; that limit is recorded in `optimization/README.md`.
