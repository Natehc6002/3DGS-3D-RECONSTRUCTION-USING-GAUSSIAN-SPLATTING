"""Run Brush training on a project folder.

Project layout (inputs never modified):
  <project>/colmap/undistorted   Brush input (images/ + sparse/)
  <project>/checkpoints/         periodic PLY exports (created)
  <project>/scenes/original/     final PLY copied here (created)
  <project>/logs/brush.log       run info log

Brush is a GUI exe and prints nothing to the console, so progress is
shown by watching the export folder for new PLY files.
"""
import argparse
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="Brush training runner")
    ap.add_argument("--project", required=True, help="project folder")
    ap.add_argument("--brush", required=True, help="path to brush_app.exe")
    ap.add_argument("--steps", type=int, default=30000, help="training steps")
    ap.add_argument("--export-every", type=int, default=5000,
                    help="write a PLY every N steps")
    ap.add_argument("--eval-split-every", type=int, default=0,
                    help="hold out every Nth image for eval (0 = off)")
    ap.add_argument("--with-viewer", action="store_true",
                    help="open Brush viewer during training")
    args = ap.parse_args()

    project = Path(args.project).resolve()
    brush = Path(args.brush)
    dataset = project / "colmap" / "undistorted"
    out = project / "checkpoints"
    scenes = project / "scenes" / "original"
    logs = project / "logs"

    if not brush.is_file():
        print(f"ERROR: brush_app.exe not found: {brush}")
        return 2
    if not (dataset / "images").is_dir() or not (dataset / "sparse").is_dir():
        print(f"ERROR: need images/ and sparse/ in {dataset}. Run COLMAP first.")
        return 2
    if args.steps < 1 or args.export_every < 1 or args.eval_split_every < 0:
        print("ERROR: --steps and --export-every must be >= 1, "
              "--eval-split-every must be >= 0")
        return 2

    name = (f"scene_{args.steps // 1000}k_{{iter}}.ply"
            if args.steps >= 1000 else "scene_{iter}.ply")
    prefix = name.split("{iter}")[0]
    final_name = name.replace("{iter}", str(args.steps))
    if out.is_dir() and list(out.glob(prefix + "*.ply")):
        print(f"ERROR: {out} already has {prefix}*.ply. Move or delete them yourself.")
        return 2
    if (scenes / final_name).exists():
        print(f"ERROR: {scenes / final_name} already exists. Move or delete it yourself.")
        return 2

    out.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(brush), str(dataset),
        "--total-steps", str(args.steps),
        "--export-every", str(args.export_every),
        "--export-path", str(out),
        "--export-name", name,
    ]
    if args.eval_split_every:
        cmd += ["--eval-split-every", str(args.eval_split_every)]
    if args.with_viewer:
        cmd.append("--with-viewer")

    start = time.time()
    seen = set(p.name for p in out.glob("*.ply"))
    initial_count = len(seen)

    with open(logs / "brush.log", "a", encoding="utf-8") as log:
        log.write(f"\n[{datetime.now():%Y-%m-%d %H:%M:%S}] $ "
                  f"{subprocess.list2cmdline(cmd)}\n")
        log.flush()

        def report_new():
            for p in sorted(out.glob("*.ply")):
                if p.name not in seen:
                    seen.add(p.name)
                    msg = (f"[{time.time() - start:6.0f}s] export: {p.name} "
                           f"({p.stat().st_size / 1e6:.1f} MB)")
                    print(msg)
                    log.write(msg + "\n")
                    log.flush()

        print("Training started. Brush draws its own progress bar (not shown with --with-viewer).")
        print(f"Watching {out} for exports...")
        proc = subprocess.Popen(cmd)
        try:
            while proc.poll() is None:
                time.sleep(5)
                report_new()
        except KeyboardInterrupt:
            proc.terminate()
            log.write("interrupted by user\n")
            print("Interrupted.")
            return 130
        rc = proc.returncode
        report_new()
        log.write(f"exit code {rc}, {time.time() - start:.0f}s\n")

    if rc != 0:
        print(f"ERROR: Brush exited with code {rc}. See {logs / 'brush.log'}")
        return rc
    if len(seen) == initial_count:
        print(f"ERROR: Brush finished but wrote no new PLY in {out}")
        return 3

    final_src = out / final_name
    if not final_src.is_file():
        plys = sorted(out.glob(prefix + "*.ply"), key=lambda p: p.stat().st_mtime)
        newest = plys[-1]
        if int(newest.stem.rsplit("_", 1)[1]) != args.steps:
            print(f"ERROR: training stopped before step {args.steps}. "
                  f"Last export: {newest.name}. Nothing copied to {scenes}.")
            return 4
        final_src = newest
    scenes.mkdir(parents=True, exist_ok=True)
    shutil.copy2(final_src, scenes / final_src.name)

    print(f"OK. Done in {time.time() - start:.0f}s. "
          f"Final scene: {scenes / final_src.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
