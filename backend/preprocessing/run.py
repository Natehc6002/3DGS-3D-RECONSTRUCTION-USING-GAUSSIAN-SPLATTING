"""Prepare input images for COLMAP.

Project layout (existing images are never overwritten or modified):
  <project>/input/video/                 optional source video
  <project>/input/images/                images to validate (or frames go here)
  <project>/input/images.tmp/            scratch while extracting, removed after
  <project>/logs/preprocessing.log       run info log (full blur list)

Modes:
  default       validate every image in input/images
  --video FILE  extract frames into input/images, then validate

Frames are written to images.tmp first and moved into input/images only when
extraction finishes. A killed run leaves no half-filled images folder.
Validation checks: readable, resolution mix, blur (warning only), count.
"""
import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

EXTS = {".jpg", ".jpeg", ".png"}


def read_image(path):
    # imread fails on non-ASCII Windows paths, so decode from bytes
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def extract_frames(video, dest, fps, max_frames, say):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        say(f"ERROR: cannot open video {video}")
        return None
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    step = max(1, round(src_fps / fps))
    say(f"video: {src_fps:.1f} fps source, keeping every {step}th frame")
    dest.mkdir(parents=True, exist_ok=True)
    idx = saved = 0
    capped = False
    while True:
        got, frame = cap.read()
        if not got:
            break
        if idx % step == 0:
            enc_ok, buf = cv2.imencode(".jpg", frame,
                                       [cv2.IMWRITE_JPEG_QUALITY, 95])
            if enc_ok:
                buf.tofile(str(dest / f"frame_{saved + 1:05d}.jpg"))
                saved += 1
                if max_frames and saved >= max_frames:
                    capped = True
                    break
        idx += 1
    cap.release()
    say(f"extracted {saved} frames")
    if capped and total and idx + step < total:
        say(f"WARN: stopped at --max-frames {max_frames} after "
            f"{(idx + 1) / src_fps:.1f}s of {total / src_fps:.1f}s. "
            "Rest of the video was not read. Lower --fps or raise "
            "--max-frames.")
    return saved


def validate(images, blur_threshold, say):
    files = sorted(p for p in images.iterdir() if p.suffix.lower() in EXTS)
    if not files:
        say(f"ERROR: no .jpg/.jpeg/.png images in {images}")
        return 3

    bad, sizes, blurry = [], {}, []
    for p in files:
        img = read_image(p)
        if img is None:
            bad.append(p.name)
            continue
        h, w = img.shape[:2]
        sizes[(w, h)] = sizes.get((w, h), 0) + 1
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        score = cv2.Laplacian(gray, cv2.CV_64F).var()
        if score < blur_threshold:
            blurry.append((p.name, score))

    say(f"images: {len(files)} found, {len(files) - len(bad)} readable")
    for (w, h), n in sorted(sizes.items(), key=lambda kv: -kv[1]):
        say(f"  {w}x{h}: {n} images")
    if len(sizes) > 1:
        say("WARN: mixed resolutions, check your COLMAP camera settings.")
    if blurry:
        pct = 100 * len(blurry) / len(files)
        names = ", ".join(n for n, _ in blurry[:10])
        more = " ... (full list in logs/preprocessing.log)" \
            if len(blurry) > 10 else ""
        say(f"WARN: {len(blurry)} images ({pct:.0f}%) below blur threshold "
            f"{blur_threshold:g}: {names}{more}")
        if len(blurry) > 10:
            say("blur scores of all flagged images:", echo=False)
            for n, s in blurry:
                say(f"  {n} {s:.1f}", echo=False)
    if len(files) < 20:
        say("WARN: fewer than 20 images. Reconstruction may fail.")
    if bad:
        say(f"ERROR: unreadable images: {', '.join(bad)}")
        return 3
    say("OK. Images ready for COLMAP.")
    return 0


def main():
    ap = argparse.ArgumentParser(description="Image preparation for COLMAP")
    ap.add_argument("--project", required=True, help="project folder")
    ap.add_argument("--video", help="extract frames from this video first")
    ap.add_argument("--fps", type=float, default=2.0,
                    help="frames kept per second of video")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="stop after N frames (0 = no limit)")
    ap.add_argument("--blur-threshold", type=float, default=50.0,
                    help="warn below this Laplacian variance (depends on "
                         "resolution, tune per dataset)")
    args = ap.parse_args()

    if args.fps <= 0 or args.max_frames < 0 or args.blur_threshold < 0:
        print("ERROR: --fps must be > 0, --max-frames and "
              "--blur-threshold must be >= 0")
        return 2

    project = Path(args.project).resolve()
    images = project / "input" / "images"
    tmp = project / "input" / "images.tmp"
    logs = project / "logs"
    video = Path(args.video) if args.video else None

    if video and not video.is_file():
        print(f"ERROR: video not found: {video}")
        return 2
    if not video and not images.is_dir():
        print(f"ERROR: {images} not found. Add images or use --video.")
        return 2
    if video and images.is_dir() and any(
            p.suffix.lower() in EXTS for p in images.iterdir()):
        print(f"ERROR: {images} already has images. "
              "Move or delete them yourself.")
        return 2

    logs.mkdir(parents=True, exist_ok=True)
    with open(logs / "preprocessing.log", "a", encoding="utf-8") as log:
        def say(msg, echo=True):
            if echo:
                print(msg)
            log.write(msg + "\n")
            log.flush()

        log.write(f"\n[{datetime.now():%Y-%m-%d %H:%M:%S}] "
                  f"preprocessing {project}\n")
        if video:
            if tmp.exists():
                # scratch folder from an earlier killed run, ours to remove
                shutil.rmtree(tmp)
            try:
                saved = extract_frames(video, tmp, args.fps,
                                       args.max_frames, say)
                if not saved:
                    say("ERROR: no frames extracted")
                    return 3
                images.mkdir(parents=True, exist_ok=True)
                for f in sorted(tmp.iterdir()):
                    f.replace(images / f.name)
                say(f"frames saved to {images}")
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
        return validate(images, args.blur_threshold, say)


if __name__ == "__main__":
    sys.exit(main())