"""Run COLMAP on a project folder, then undistort for Brush.

Project layout (inputs never modified):
  <project>/input/images        photos (read only)
  <project>/colmap/             COLMAP output (created)
  <project>/colmap/undistorted  Brush input (images/ + sparse/)
  <project>/logs/colmap.log     full log
"""
import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd, log):
    log.write("\n$ " + " ".join(cmd) + "\n")
    log.flush()
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, errors="replace",
    )
    for line in proc.stdout:
        print(line, end="")
        log.write(line)
    proc.wait()
    log.flush()
    return proc.returncode


def main():
    ap = argparse.ArgumentParser(description="COLMAP + undistort runner")
    ap.add_argument("--project", required=True, help="project folder")
    ap.add_argument("--colmap", required=True, help="path to colmap.exe")
    ap.add_argument("--quality", default="medium",
                    choices=["low", "medium", "high", "extreme"])
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
    if (out / "sparse").exists():
        print(f"ERROR: {out / 'sparse'} already exists. "
              "Move or delete it yourself. Script never overwrites.")
        return 2

    out.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)

    with open(logs / "colmap.log", "a", encoding="utf-8") as log:
        print("Step 1/2: reconstruction")
        rc = run([
            str(colmap), "automatic_reconstructor",
            "--workspace_path", str(out),
            "--image_path", str(images),
            "--data_type", "individual",
            "--single_camera", "1",
            "--quality", args.quality,
            "--dense", "0",
            "--use_gpu", "1",
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

    print(f"OK. Brush input: {out / 'undistorted'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
