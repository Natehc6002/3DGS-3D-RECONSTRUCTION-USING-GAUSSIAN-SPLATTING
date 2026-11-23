# Sidecar contract (`<name>.normalize.json`)

Every normalisation writes a mandatory sidecar JSON next to its output
PLY. The sidecar is the only thing that makes the transform reversible:
it records exactly what was done (centre, uniform scale, optional
upright rotation), how confident each automatic choice was, and the
verification that the written file round-trips.

Producers: `optimization/normalize/normalize.py` (CLI and the
`normalize_file()` library entry point the pipeline calls).
Implementation: `optimization/normalize/core.py`
(`build_sidecar`, `sidecar_to_transform`, `undo_vertex`).

## Naming and lifecycle

- Default path: `<output stem>.normalize.json` next to the output PLY
  (`default_sidecar_path()`); override with `--sidecar`.
- Pipeline: `<final stem>_norm.normalize.json`; reported at
  `outputs.normalize_sidecar` and `normalize.sidecar_path` in the run
  report. Normalize plan warnings are propagated into the pipeline
  `warnings` list prefixed `normalize: `.
- Written **only on success**: paths are validated first, the transform
  is applied, and the sidecar round trip must stay within
  `ROUNDTRIP_REL_TOLERANCE` (worst relative error <= 1e-5) or nothing
  is written (exit 3).
- Never overwritten: an existing sidecar or output PLY is refused
  (exit 2). The input PLY is never modified.

## Transform convention

The `convention` string inside the sidecar states it verbatim:

```text
p_out = s * R * (p_in - c); log(scale)_out = log(scale)_in + ln(s);
rot_out = q_R (x) rot_in (unit quaternions, rot_0 = w); SH rotated so
colour_out(d) = colour_in(R^T d); opacity and f_dc unchanged; matrix
maps p_in -> p_out, matrix_inv maps back
```

In fields: `c` = `center`, `s` = `s`, `R` = `R` (identity when no
upright rotation was applied). Consequences:

- positions: one rigid + uniform-scale move;
- `scale_0..2` (log space): every axis shifts by `ln(s)`, so
  anisotropy is preserved;
- `rot_0..3`: composed with `q_R` on the left (`rot_0` = w);
- SH coefficients: rotated by `R` so the rendered colour of a direction
  is unchanged; verified against the Brush SH basis when `R != I`
  (`verification.sh_rotation_*`);
- `opacity`, `f_dc_*`: byte-identical.

## Fields

| Field | Type | Meaning |
|---|---|---|
| `tool` | str | always `optimization.normalize` |
| `tool_version` | str | contract version, currently `1` |
| `created_at` | str | UTC ISO timestamp of the run |
| `source`, `output` | str | input and output PLY paths (input is untouched) |
| `convention` | str | the transform formula above (human/machine readable) |
| `unit` | str | what `s` achieved, e.g. `subject radius p90 = 1 in the output (input 15.1283 x scale 0.0661...)` |
| `matrix` | 4x4 list | maps `p_in` -> `p_out` (`A = s R`, `b = -A c`) |
| `matrix_inv` | 4x4 list | maps `p_out` -> `p_in` (used by `--undo`) |
| `R`, `R_inv` | 3x3 list | upright rotation and its transpose (identity if none) |
| `t` | 3 list | translation part `b` of `matrix` |
| `s` | float | uniform scale (> 0) |
| `center` | 3 list | subject centre `c` subtracted before scaling |
| `up` | 3 list or null | estimated up direction used (null = no rotation) |
| `up_angle_to_brush_y_deg` | float or null | angle the up direction had to rotate (null = no rotation) |
| `methods` | dict | per-step method strings: `center`, `scale`, and `upright` when applied (e.g. `subject density centre (k-NN of compact bright Gaussians, dominant plane excluded)`, `robust centre (subject confidence too low)`) |
| `confidence` | dict | 0..1 confidence per automatic choice (`center`, `scale`, `upright`); null when not applicable |
| `warnings` | list[str] | plan warnings (e.g. low-confidence fallbacks); the pipeline re-emits them as `normalize: ...` |
| `verification` | dict | see below |

`verification`:

| Key | Meaning |
|---|---|
| `sh_rotation_checked` | whether the SH rotation was numerically verified (only when `R != I`) |
| `sh_rotation_max_error`, `sh_tolerance` | max error and tolerance (1e-5) against the Brush SH basis (Sloan JCGT 2013, `brush-render/src/kernels/sh.rs`) |
| `sh_basis` | where the basis is taken from |
| `roundtrip_max_error` | per-attribute max absolute error of apply-then-undo (`position`, `scale`, `rot`, `opacity`, `f_dc`, `f_rest`) |
| `roundtrip_worst_relative` | worst relative error; the write is refused above `ROUNDTRIP_REL_TOLERANCE` (1e-5) |

## Undo

```powershell
python -m optimization.normalize.normalize --undo <sidecar.json> `
    --input <normalized.ply> --output <restored.ply>
```

`sidecar_to_transform()` validates the minimum field set (`matrix_inv`
4x4, `s` finite > 0, `R` 3x3) before anything is applied; the inverse
reverses scale (`-ln s`), rotation (conjugate quaternion / `R` applied
backwards) and translation (`matrix_inv`). Output rules are identical
to the forward path: never overwrite, input untouched.

## Evidence rules

Every automatic centre/up/scale records its `methods` entry and
`confidence`; low confidence either falls back (centre -> robust
centre) or warns (up sign may be flipped) and both appear in
`warnings`. Manual inputs (`--center`, `--up-axis`, `--scale`,
`--plane`-style overrides) always win and are recorded in `methods`.
No field may claim a confidence that was not computed - absent evidence
reads `null` / `n/a`, never a guessed number.
