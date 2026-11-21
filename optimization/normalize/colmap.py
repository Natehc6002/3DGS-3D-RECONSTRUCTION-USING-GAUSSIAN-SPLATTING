"""COLMAP sparse model reader (no external dependency).

Reads `cameras.bin|txt` and `images.bin|txt` from a COLMAP sparse
folder so `--colmap-dir` can supply capture-side evidence for the
normalisation:

    - subject centre = point closest to all camera viewing axes
      (least-squares ray intersection)
    - orientation    = average camera up direction

Only the pose part of the format is used; 2D/3D points are skipped.
Binary parsing follows the COLMAP format specification (little-endian,
`uint64` counts, null-terminated name strings). Text parsing follows
the documented two-line-per-image layout of `images.txt`.

Convention (COLMAP): image x right, y down, z forward;
`x_cam = R * x_world + t`, quaternion (qw, qx, qy, qz) is world-to-
camera. Camera centre C = -R^T t; viewing axis = R^T (0,0,1); camera
up = R^T (0,-1,0) (image y points down).
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

# COLMAP camera model id -> number of parameters.
MODEL_PARAM_COUNTS = {
    0: 3,    # SIMPLE_PINHOLE
    1: 4,    # PINHOLE
    2: 4,    # SIMPLE_RADIAL
    3: 5,    # RADIAL
    4: 8,    # OPENCV
    5: 8,    # OPENCV_FISHEYE
    6: 12,   # FULL_OPENCV
    7: 5,    # FOV
    8: 4,    # SIMPLE_RADIAL_FISHEYE
    9: 5,    # RADIAL_FISHEYE
    10: 12,  # THIN_PRISM_FISHEYE
}
MIN_CAMERAS_FOR_CENTRE = 2
# A (nearly) parallel ray set cannot pin a point: refuse instead of
# returning a garbage intersection far outside the scene.
MIN_AXIS_SPREAD_RATIO = 1e-9


def _pick_files(folder):
    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError(f"COLMAP folder not found: {folder}")
    cams = _first_existing(folder, ("cameras.bin", "cameras.txt"))
    imgs = _first_existing(folder, ("images.bin", "images.txt"))
    if cams is None and imgs is None:
        raise ValueError(
            f"no COLMAP sparse files (cameras/images .bin or .txt) in: {folder}")
    return cams, imgs


def _first_existing(folder, names):
    for name in names:
        p = folder / name
        if p.is_file():
            return p
    return None


def read_sparse(folder):
    """Return {"cameras": [...], "images": [...], "files": {...}}."""
    cams_path, imgs_path = _pick_files(folder)
    cameras = []
    images = []
    if cams_path is not None:
        cameras = (_read_cameras_bin(cams_path)
                   if cams_path.suffix == ".bin"
                   else _read_cameras_txt(cams_path))
    if imgs_path is not None:
        images = (_read_images_bin(imgs_path)
                  if imgs_path.suffix == ".bin"
                  else _read_images_txt(imgs_path))
    if not images:
        raise ValueError(f"no images found in COLMAP folder: {folder}")
    return {"cameras": cameras, "images": images,
            "files": {"cameras": str(cams_path) if cams_path else None,
                      "images": str(imgs_path) if imgs_path else None}}


def _read_cameras_txt(path):
    cameras = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"bad camera line in {path.name}: {line[:60]}")
        cameras.append({"camera_id": int(parts[0]), "model": parts[1],
                        "width": int(parts[2]), "height": int(parts[3]),
                        "params": [float(x) for x in parts[4:]]})
    return cameras


def _read_cameras_bin(path):
    cameras = []
    with open(path, "rb") as f:
        (count,) = struct.unpack("<Q", f.read(8))
        for _ in range(count):
            header = f.read(24)   # u32 id + u32 model + u64 w + u64 h
            if len(header) < 24:
                raise ValueError(f"truncated {path.name}")
            camera_id, model_id, width, height = struct.unpack("<IIQQ", header)
            if model_id not in MODEL_PARAM_COUNTS:
                raise ValueError(f"unknown COLMAP camera model id {model_id}")
            n_params = MODEL_PARAM_COUNTS[model_id]
            raw = f.read(8 * n_params)
            if len(raw) < 8 * n_params:
                raise ValueError(f"truncated {path.name}")
            params = list(struct.unpack(f"<{n_params}d", raw))
            cameras.append({"camera_id": camera_id, "model": str(model_id),
                            "width": width, "height": height,
                            "params": params})
    return cameras


def _read_images_txt(path):
    images = []
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    expect_pose = True
    for line in lines:
        if line.startswith("#"):
            continue
        if expect_pose:
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 10:
                raise ValueError(
                    f"bad image pose line in {path.name}: {line[:60]}")
            images.append({
                "image_id": int(parts[0]),
                "q": [float(x) for x in parts[1:5]],   # qw qx qy qz
                "t": [float(x) for x in parts[5:8]],
                "camera_id": int(parts[8]),
                "name": " ".join(parts[9:]),
            })
            expect_pose = False
        else:
            # POINTS2D line (may be empty); its content is not needed.
            expect_pose = True
    return images


def _read_images_bin(path):
    images = []
    with open(path, "rb") as f:
        (count,) = struct.unpack("<Q", f.read(8))
        for _ in range(count):
            head = f.read(64)   # u32 id + 7 f64 + u32 camera = 64 bytes
            if len(head) < 64:
                raise ValueError(f"truncated {path.name}")
            image_id = struct.unpack("<I", head[:4])[0]
            vals = struct.unpack("<7d", head[4:60])
            camera_id = struct.unpack("<I", head[60:64])[0]
            name_bytes = bytearray()
            while True:
                ch = f.read(1)
                if not ch:
                    raise ValueError(f"truncated {path.name}")
                if ch == b"\x00":
                    break
                name_bytes += ch
            (n_pts,) = struct.unpack("<Q", f.read(8))
            # Skip points: x f64, y f64, point3D_id i64 = 24 bytes each.
            if n_pts:
                skipped = f.read(24 * n_pts)
                if len(skipped) < 24 * n_pts:
                    raise ValueError(f"truncated {path.name}")
            images.append({"image_id": image_id,
                           "q": list(vals[0:4]),
                           "t": list(vals[4:7]),
                           "camera_id": camera_id,
                           "name": name_bytes.decode("utf-8",
                                                     errors="replace")})
    return images


def quat_to_matrix(q):
    """(qw, qx, qy, qz) world-to-camera rotation matrix."""
    w, x, y, z = [float(v) for v in q]
    n = w * w + x * x + y * y + z * z
    if n <= 0:
        raise ValueError("COLMAP image has a zero quaternion")
    s = 2.0 / n
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - w * z), s * (x * z + w * y)],
        [s * (x * y + w * z), 1 - s * (x * x + z * z), s * (y * z - w * x)],
        [s * (x * z - w * y), s * (y * z + w * x), 1 - s * (x * x + y * y)],
    ], dtype=np.float64)


def poses(images):
    """Camera centres, world viewing axes and world up vectors."""
    centers, dirs, ups = [], [], []
    for img in images:
        r = quat_to_matrix(img["q"])                 # world -> camera
        t = np.asarray(img["t"], dtype=np.float64)
        centers.append(-r.T @ t)
        dirs.append(r.T @ np.array([0.0, 0.0, 1.0]))
        ups.append(r.T @ np.array([0.0, -1.0, 0.0]))
    return (np.array(centers), np.array(dirs), np.array(ups))


def intersect_rays(centers, dirs):
    """Least-squares point closest to all viewing rays.

    Minimizes sum_i ||(I - v_i v_i^T)(x - C_i)||^2. Confidence is
    1 - mean perpendicular ray offset / mean camera distance (1 = every
    ray passes exactly through the solution). Raises ValueError when
    there are too few cameras or the axes are (nearly) parallel.
    """
    centers = np.asarray(centers, dtype=np.float64)
    dirs = np.asarray(dirs, dtype=np.float64)
    if len(centers) < MIN_CAMERAS_FOR_CENTRE:
        raise ValueError(
            f"need at least {MIN_CAMERAS_FOR_CENTRE} cameras to intersect "
            f"viewing axes (got {len(centers)})")
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("COLMAP image has a zero viewing axis")
    v = dirs / norms
    a = np.zeros((3, 3))
    b = np.zeros(3)
    for c_i, v_i in zip(centers, v):
        p = np.eye(3) - np.outer(v_i, v_i)
        a += p
        b += p @ c_i
    evals = np.linalg.eigvalsh(a)
    if evals[-1] <= 0 or evals[0] / evals[-1] < MIN_AXIS_SPREAD_RATIO:
        raise ValueError(
            "camera viewing axes are (nearly) parallel; cannot intersect "
            "them - use the PLY centre instead")
    point = np.linalg.solve(a, b)
    delta = point - centers
    along = np.sum(delta * v, axis=1)
    perp = np.linalg.norm(delta - along[:, None] * v, axis=1)
    dist = np.linalg.norm(delta, axis=1)
    mean_dist = float(np.mean(dist))
    confidence = 0.0 if mean_dist <= 0 else 1.0 - float(np.mean(perp)) / mean_dist
    confidence = float(min(1.0, max(0.0, confidence)))
    return point, confidence, {
        "mean_perpendicular_offset": float(np.mean(perp)),
        "mean_camera_distance": mean_dist,
        "cameras": int(len(centers)),
    }


def average_up(ups):
    """Mean camera up direction and mean resultant length in [0, 1]."""
    ups = np.asarray(ups, dtype=np.float64)
    if len(ups) == 0:
        raise ValueError("no camera up vectors")
    norms = np.linalg.norm(ups, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("COLMAP image has a zero up vector")
    unit = ups / norms
    mean = unit.mean(axis=0)
    resultant = float(np.linalg.norm(mean))
    if resultant <= 1e-12:
        raise ValueError(
            "camera up directions cancel out (cameras are rolled/spread "
            "evenly); no reliable orientation")
    return mean / resultant, resultant
