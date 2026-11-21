# Optimization evaluation

- Generated: 2026-11-21T12:44:48+00:00
- Original: `projects\figure_01\scenes\original\figure_30k_30000.ply`
- Quality metrics (PSNR, SSIM, FPS) are not measured anywhere in this repository yet.

| variant | gaussians | size (MB) | ratio | time (s) | coverage proxy (voxels) | Route B round-trip max abs diff | Brush load | PSNR | SSIM | FPS |
|---|---|---|---|---|---|---|---|---|---|---|
| original | 258309 | 60.96 | 1.000 | n/a | n/a (no removals) | n/a | n/a (Brush export) | not measured | not measured | not measured |
| baseline_prune | 178344 | 42.09 | 1.448 | 0.3 | voxels emptied 610; over 50% loss 910; max loss 1.000 | n/a | not tested | not measured | not measured | not measured |
| smart_default | 182489 | 43.07 | 1.415 | 1.1 | voxels emptied 0; over 50% loss 0; max loss 0.300 | n/a | not tested | not measured | not measured | not measured |
| smart_shd2 | 182489 | 27.74 | 2.198 | 13.1 | n/a (layout differs from original) | n/a | opens ok (no load error); GPU: Intel iGPU (Vulkan) | not measured | not measured | not measured |
| smart_shd1 | 182489 | 16.79 | 3.631 | 12.7 | n/a (layout differs from original) | n/a | not tested | not measured | not measured | not measured |
| smart_shd0 | 182489 | 10.22 | 5.965 | 12.8 | n/a (layout differs from original) | n/a | not tested | not measured | not measured | not measured |
| route_b_decoded | 182489 | 43.07 | 1.415 | 13.7 | n/a (not an exact subset of original) | 0.00547457 | not tested | not measured | not measured | not measured |

## Notes

- Size ratio is original size / variant size (higher = smaller).
- Coverage proxy is voxel occupancy loss versus the original (geometric proxy only, see optimization/pruning/coverage_check.py).
- Route B round-trip is the max absolute difference over all PLY properties, source vs decoded PLY.
- Brush load status comes from manual visual checks; "not tested" means nobody has opened that file in Brush yet.
- The .npz archive itself is not readable by Brush - only the decoded PLY in this table is.
