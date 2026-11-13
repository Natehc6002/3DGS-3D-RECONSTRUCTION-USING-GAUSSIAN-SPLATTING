"""Size-reduction routes for Brush-exported PLY scenes.

Route A: sh_truncate.py   - SH degree truncation (fewer properties)
Route B: quantize.py / dequantize.py - uint8/uint16 archive (.npz)
Route C: gzip_ply.py      - gzip baseline of the plain PLY

Brush reads only float32 PLYs; the .npz archive is for storage and
transfer, dequantize.py writes a normal float32 PLY back.
"""
