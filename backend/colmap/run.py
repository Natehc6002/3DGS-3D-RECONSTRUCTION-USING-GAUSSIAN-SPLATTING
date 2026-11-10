"""Run COLMAP on a project folder, then undistort for Brush.

Project layout (inputs never modified):
  <project>/input/images        photos (read only)
  <project>/colmap/             COLMAP output (created)
  <project>/colmap/undistorted  Brush input (images/ + sparse/)
  <project>/logs/colmap.log     full log

After reconstruction the runner reports how many images each sparse model
holds and warns if the model Brush will use (sparse/0) is incomplete.
"""
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

EXTS = {".jpg", ".jpeg", ".png"}
MIN_REGISTERED = 0.8


def colmap_environment(executable):
    """Match the PATH and Qt plugin setup from the Windows COLMAP.bat launcher."""
    exe = Path(executable).resolve()
    bin_dir = exe.parent
    package_root = bin_dir.parent if bin_dir.name.lower() == "bin" else bin_dir
    env = os.environ.copy()
    env["PATH"] = str(bin_dir) + os.pathsep + env.get("PATH", "")
    plugins = package_root / "plugins"
    if plugins.is_dir():
        current = env.get("QT_PLUGIN_PATH", "")
        env["QT_PLUGIN_PATH"] = str(plugins) + (os.pathsep + current if current else "")
    return env


def run(cmd, log):
    log.write("\n$ " + " ".join(cmd) + "\n")
    log.flush()
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, errors="replace", env=colmap_environment(cmd[0]),
    )
    for line in proc.stdout:
        print(line, end="")
        log.write(line)
    proc.wait()
    log.flush()
    return proc.returncode


def registered_images(colmap, model):
    r = subprocess.run([str(colmap), "model_analyzer", "--path", str(model)],
                       capture_output=True, text=True, errors="replace",
                       env=colmap_environment(colmap))
    m = re.search(r"Registered images:\s*(\d+)", r.stdout + r.stderr)
    return int(m.group(1)) if m else None


def check_models(colmap, out, images, note):
    sparse = out / "sparse"
    models = sorted((p for p in sparse.iterdir()
                     if p.is_dir() and p.name.isdigit()),
                    key=lambda p: int(p.name))
    total = sum(1 for p in images.iterdir() if p.suffix.lower() in EXTS)
    counts = {p.name: registered_images(colmap, p) for p in models}

    note(f"sparse models: {len(models)}")
    for name, n in counts.items():
        if n is None:
            note(f"  sparse/{name}: could not read image count")
            continue
        used = "  <- used by Brush" if name == "0" else ""
        note(f"  sparse/{name}: {n} of {total} images "
             f"({100 * n / total:.0f}%){used}")

    main = counts.get("0")
    if main is None:
        return
    if len(models) > 1:
        note("WARN: COLMAP split the scene into several models. Brush uses "
             "sparse/0 only, so the scene covers part of the images. Check "
             "where the camera path breaks (turn in place, plain surface, "
             "fast move).")
    if any(n and n > main for n in counts.values()):
        note("WARN: another model has more images than sparse/0.")
    if main / total < MIN_REGISTERED:
        note(f"WARN: only {100 * main / total:.0f}% of images are in "
             f"sparse/0 (below {MIN_REGISTERED:.0%}). Expect holes in the "
             "scene. Consider reshooting.")


def main():
    ap = argparse.ArgumentParser(description="COLMAP + undistort runner")
    ap.add_argument("--project", required=True, help="project folder")
    ap.add_argument("--colmap", required=True, help="path to colmap.exe")
    ap.add_argument("--quality", default="medium",
                    choices=["low", "medium", "high", "extreme"])
    ap.add_argument("--use-gpu", type=int, choices=[0, 1], default=1,
                    help="use CUDA for feature extraction/matching (1) or CPU (0)")
    ap.add_argument("--data-type", default="individual",
                    choices=["individual", "video", "internet"],
                    help="COLMAP data type. 'video' is meant for ordered "
                         "frames from one clip (default: individual)")
    args = ap.parse_args()

    project = Path(args.project).resolve()
    colmap = Path(args.colmap)
    images = project / "input" / "images"
    out = project / "colmap"
    logs = project / "logs"

    if not colmap.is_file():
        print(f"ERROR: colmap.exe not found: {colmap}")
        return 2
    if not images.is_dir() or not any(images.iterdir()):
        print(f"ERROR: no images in {images}")
        return 2
    sparse = out / "sparse"
    if sparse.exists() and any(sparse.iterdir()):
        print(f"ERROR: {sparse} already contains a reconstruction. "
              "Move or delete it yourself. Script never overwrites.")
        return 2
    if (out / "database.db").exists():
        print(f"ERROR: {out / 'database.db'} already exists. "
              "Move or delete it yourself. Script never overwrites.")
        return 2
    undistorted = out / "undistorted"
    if undistorted.exists() and any(undistorted.iterdir()):
        print(f"ERROR: {undistorted} already contains an undistorted dataset. "
              "Move it aside yourself. Script never overwrites.")
        return 2

    out.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)

    with open(logs / "colmap.log", "a", encoding="utf-8") as log:
        def note(msg):
            print(msg)
            log.write(msg + "\n")
            log.flush()

        print("Step 1/2: reconstruction")
        rc = run([
            str(colmap), "automatic_reconstructor",
            "--workspace_path", str(out),
            "--image_path", str(images),
            "--data_type", args.data_type,
            "--single_camera", "1",
            "--quality", args.quality,
            "--dense", "0",
            "--use_gpu", str(args.use_gpu),
        ], log)
        if rc != 0:
            print(f"ERROR: reconstruction failed (exit {rc}). See {logs / 'colmap.log'}")
            return rc

        model = out / "sparse" / "0"
        needed = ["cameras.bin", "images.bin", "points3D.bin"]
        if not all((model / n).is_file() for n in needed):
            print(f"ERROR: no valid model in {model}")
            return 3

        print("Step 2/2: undistort for Brush")
        rc = run([
            str(colmap), "image_undistorter",
            "--image_path", str(images),
            "--input_path", str(model),
            "--output_path", str(out / "undistorted"),
            "--output_type", "COLMAP",
        ], log)
        if rc != 0:
            print(f"ERROR: undistort failed (exit {rc}). See {logs / 'colmap.log'}")
            return rc

        check_models(colmap, out, images, note)

    print(f"OK. Brush input: {out / 'undistorted'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
