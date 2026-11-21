"""Scene normalisation for Brush-exported PLY scenes.

normalize.py  - CLI: centre, scale, optional upright rotation, sidecar
core.py       - plan building, attribute transform, sidecar, round trip
sh_rotate.py  - SH coefficient rotation on Brush's own SH basis
colmap.py     - dependency-free COLMAP sparse (cameras/images) reader

Every transform writes a mandatory sidecar JSON (4x4 matrix + inverse,
methods, confidences, verification) so the desktop app can map between
original and normalized coordinates and undo the transform.
"""
