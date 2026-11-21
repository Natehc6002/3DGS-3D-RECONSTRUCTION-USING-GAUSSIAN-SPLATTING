# Brush GPU selection (make Brush use the NVIDIA RTX 4060)

Test machine: Windows, Brush 0.3.0, NVIDIA GeForce RTX 4060 Laptop GPU
(driver 32.0.15.8186) + Intel(R) RaptorLake-S Mobile Graphics
Controller (driver 101.7026). Every fact below is from this machine;
nothing is extrapolated to other GPUs or Brush versions.

## What actually works (evidence)

| attempt | result |
|---|---|
| Windows per-app preference `HKCU\Software\Microsoft\DirectX\UserGpuPreferences` = `GpuPreference=2;` for `C:\Tools\Brush\brush_app.exe` (already set on this machine) | **did NOT switch the adapter** - Brush still reported the Intel integrated GPU. The registry value alone is not enough. |
| `WGPU_ADAPTER_NAME=NVIDIA` / `WGPU_POWER_PREF=high` hints (old launcher) | **did NOT switch the adapter** either - Brush reported Intel. |
| `$env:VK_DRIVER_FILES` = NVIDIA Vulkan ICD json **together with** `$env:WGPU_BACKEND = "vulkan"` | **worked** - Brush used the NVIDIA GPU (reported by Chetan I S). |
| DirectX 12 backend (`WGPU_BACKEND=d3d12`) | **crashes** Brush 0.3.0 on this machine (shader compile error, FXC X3695). Only Vulkan works. |

**NVIDIA run confirmed by user, Stats panel text: Name `<...>`, Type
`<...>`** (verbatim as reported by the user; the exact panel text was
not transcribed into this document - re-check it yourself with the
verification steps below whenever you want to be sure).

The ICD json found on this machine:

```text
C:\Windows\System32\DriverStore\FileRepository\nvlt.inf_amd64_45df1d07baab9850\nv-vk64.json
```

**The `FileRepository` folder name changes after every NVIDIA driver
update**, so nothing hard-codes it: `open_in_brush.ps1` searches
`C:\Windows\System32\DriverStore\FileRepository` for `nv-vk64.json` by
name and takes the newest one by `LastWriteTime`.

## Pitfalls

- A **placeholder or wrong path** in `VK_DRIVER_FILES` makes Brush fail
  at startup with:
  `WGPU error: Failed to create surface for any enabled backend`
  (seen once). The launcher validates that the json exists before it
  launches anything.
- Setting only the registry preference (or only the `WGPU_*` hints) and
  assuming the GPU changed is the other trap - the only reliable check
  is Brush's own **Stats panel** (below).

## `open_in_brush.ps1`

```powershell
.\optimization\tools\open_in_brush.ps1 -Ply <path\to\file.ply>            # -Gpu auto (default)
.\optimization\tools\open_in_brush.ps1 -Ply <path> -Gpu nvidia            # require the NVIDIA ICD
.\optimization\tools\open_in_brush.ps1 -Ply <path> -Gpu intel             # default adapter, never touch VK_DRIVER_FILES
.\optimization\tools\open_in_brush.ps1 -Ply <path> -PrintOnly             # dry run: print, do not launch
.\optimization\tools\open_in_brush.ps1 -Ply <path> -BrushExe C:\Tools\Brush\brush_app.exe
```

- `-Gpu auto|nvidia|intel` (default `auto`):
  - `nvidia`: find the newest `nv-vk64.json`, set `VK_DRIVER_FILES` to it
    for the Brush **child process only**, together with
    `WGPU_BACKEND=vulkan`; exit 2 with a clean one-line error if the json
    is not found.
  - `auto`: same when the json exists; otherwise print a clear warning
    and fall back to the default adapter (no `VK_DRIVER_FILES`).
  - `intel`: never sets `VK_DRIVER_FILES` (default adapter); still sets
    `WGPU_BACKEND=vulkan` because D3D12 crashes Brush 0.3.0 here.
- The environment is set for the child only: the script saves your
  session's `VK_DRIVER_FILES`/`WGPU_BACKEND` and restores them after
  Brush exits.
- Expected failures (missing PLY, missing Brush, missing json) print one
  `ERROR: ...` line and exit 2 - no stack traces.

### Real `-PrintOnly` output (this machine)

```text
=== -Gpu auto ===
GPU mode:        auto
VK_DRIVER_FILES: C:\Windows\System32\DriverStore\FileRepository\nvlt.inf_amd64_45df1d07baab9850\nv-vk64.json  (child only)
WGPU_BACKEND:    vulkan  (child only)
Command:         "C:\Tools\Brush\brush_app.exe" --with-viewer "C:\Users\DARSHAN I S\Desktop\3dgs_studio\3DGS-3D-RECONSTRUCTION-USING-GAUSSIAN-SPLATTING\projects\figure_01\scenes\quantized\fig_30k_smart_shd2.ply"
(dry run - Brush not launched)
exit=0

=== -Gpu nvidia ===
GPU mode:        nvidia
VK_DRIVER_FILES: C:\Windows\System32\DriverStore\FileRepository\nvlt.inf_amd64_45df1d07baab9850\nv-vk64.json  (child only)
WGPU_BACKEND:    vulkan  (child only)
Command:         "C:\Tools\Brush\brush_app.exe" --with-viewer "C:\Users\DARSHAN I S\Desktop\3dgs_studio\3DGS-3D-RECONSTRUCTION-USING-GAUSSIAN-SPLATTING\projects\figure_01\scenes\quantized\fig_30k_smart_shd2.ply"
(dry run - Brush not launched)
exit=0

=== -Gpu intel ===
GPU mode:        intel
VK_DRIVER_FILES: (not set - default adapter)  (child only)
WGPU_BACKEND:    vulkan  (child only)
Command:         "C:\Tools\Brush\brush_app.exe" --with-viewer "C:\Users\DARSHAN I S\Desktop\3dgs_studio\3DGS-3D-RECONSTRUCTION-USING-GAUSSIAN-SPLATTING\projects\figure_01\scenes\quantized\fig_30k_smart_shd2.ply"
(dry run - Brush not launched)
exit=0

=== nonexistent PLY ===
ERROR: PLY not found: C:\no\such\file.ply
exit=2
```

Also checked on this machine: missing Brush exe ->
`ERROR: Brush executable not found: C:\no\brush.exe` (exit 2); a
non-`.ply` input -> `ERROR: Not a .ply file: requirements.txt` (exit 2);
`-Gpu nvidia` with no json anywhere -> `ERROR: nv-vk64.json not found
under ...` (exit 2); `-Gpu auto` with no json -> `WARNING: nv-vk64.json
not found ...; falling back to the default GPU adapter` and continues
(exit 0).

## How to verify (do this after every change)

1. Open a PLY with the launcher:
   ```powershell
   .\optimization\tools\open_in_brush.ps1 -Ply <path\to\file.ply> -Gpu nvidia
   ```
2. In Brush, open the **Stats** panel:
   - GPU Name must contain **NVIDIA**, Type must be **DiscreteGpu**.
   - If it shows Intel / `IntegratedGpu`, record exactly what it shows;
     the selection did not work for that launch.
3. Optional cross-check: Task Manager -> Performance -> GPU - the NVIDIA
   graph does the work while Brush renders.
4. The launcher prints the resolved `VK_DRIVER_FILES` path and the exact
   command line; use `-PrintOnly` first if you only want to see them.

Checklist for any file you open (record the result, "not tested" until
you look): Stats panel GPU name, colour shift vs original, flat shading,
holes, floaters, load errors.

## `set_brush_gpu_preference.ps1` (per-user registry)

Writes the same per-user value as
Settings > System > Display > Graphics > "High performance":
`HKCU\Software\Microsoft\DirectX\UserGpuPreferences`, value name =
full path to `brush_app.exe`, data = `GpuPreference=2;`.
Current user only (HKCU), no admin, no HKLM, and it refuses any path
other than `C:\Tools\Brush\brush_app.exe`.

**Already applied on this machine.** The value alone did NOT switch the
adapter (table above), so the launcher does not depend on it.

```powershell
# show current value (read-only)
.\optimization\tools\set_brush_gpu_preference.ps1
# set High performance (prints the previous value first)
.\optimization\tools\set_brush_gpu_preference.ps1 -Apply
# Undo: remove only this value (the only thing -Apply ever created)
.\optimization\tools\set_brush_gpu_preference.ps1 -Undo
```

After `-Apply` (and after `-Undo`) you must **fully close Brush (all
windows) and reopen it**; Windows reads the preference when the process
starts.

## Manual steps (equivalent, no scripts)

Windows: Settings -> System -> Display -> Graphics -> add
`C:\Tools\Brush\brush_app.exe` -> Options -> "High performance" -> Save.

NVIDIA Control Panel: Manage 3D settings -> Program Settings ->
Add `brush_app.exe` -> "High-performance NVIDIA processor" -> Apply.
(Exact wording varies by driver version.)

## If the Stats panel still shows Intel

1. Confirm the launcher actually resolved an ICD json (`-PrintOnly`,
   the `VK_DRIVER_FILES` line) - a missing/placeholder path fails with
   the `Failed to create surface` error instead of silently using Intel.
2. Advanced Optimus / MUX switch: many laptops only expose the discrete
   GPU to apps when the dGPU drives the display - check your laptop
   vendor app (Lenovo Vantage, MyASUS, Armoury Crate, ...) or BIOS for
   "MUX switch" / "Discrete graphics" / "nVIDIA only" modes.
3. Update the NVIDIA driver (current on this machine: 32.0.15.8186) via
   NVIDIA GeForce Experience or the laptop vendor - then re-run the
   launcher so it picks up the new `nv-vk64.json` folder.

## Status

- `VK_DRIVER_FILES` + `WGPU_BACKEND=vulkan`: **NVIDIA run confirmed by
  user** (Stats panel text Name `<...>`, Type `<...>` as reported).
- Registry preference alone: **confirmed not sufficient** on this
  machine (Brush stayed on Intel).
- DirectX 12 backend: **crashes** Brush 0.3.0 here (FXC X3695); Vulkan
  only.
- Early Brush verdicts in `optimization/README.md` were obtained on the
  Intel integrated GPU; the GPU used per file is recorded next to each
  verdict.
