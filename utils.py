"""
utils.py
--------
Helper utilities:
  - download a YouTube / public video via yt-dlp
  - generate a simple per-frame object count chart
  - collect video metadata
"""

import cv2
import os
import subprocess
import json
import tempfile
import numpy as np


# ─────────────────────────────────────────────
# Download public video with yt-dlp
# ─────────────────────────────────────────────
def download_video(url: str, output_dir: str = "downloads") -> str:
    """
    Download a video from a public URL (YouTube, etc.) using yt-dlp.

    Parameters
    ----------
    url        : public video URL
    output_dir : folder to save the video in

    Returns
    -------
    Path to the downloaded file.

    Raises
    ------
    RuntimeError if yt-dlp fails.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Output template — yt-dlp fills in the title
    out_template = os.path.join(output_dir, "%(title).60s.%(ext)s")

    # Try high-quality stream + merge first.
    primary_cmd = [
        "yt-dlp",
        "--no-playlist",
        "-f", "bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "-o", out_template,
        "--print", "after_move:filepath",   # prints final path(s) to stdout
        url,
    ]
    primary = subprocess.run(primary_cmd, capture_output=True, text=True)

    # Collect all existing file paths reported by yt-dlp.
    reported = []
    for line in (primary.stdout or "").splitlines():
        path = line.strip()
        if path and os.path.exists(path):
            reported.append(path)

    # If merge worked, prefer a merged mp4 path.
    mp4_reported = [p for p in reported if p.lower().endswith(".mp4")]
    if mp4_reported:
        return max(mp4_reported, key=os.path.getsize)

    # Fallback when ffmpeg isn't available for merge:
    # request a single progressive file that does not need merging.
    fallback_cmd = [
        "yt-dlp",
        "--no-playlist",
        "-f", "best[ext=mp4]/best",
        "-o", out_template,
        "--print", "after_move:filepath",
        url,
    ]
    fallback = subprocess.run(fallback_cmd, capture_output=True, text=True)

    fallback_reported = []
    for line in (fallback.stdout or "").splitlines():
        path = line.strip()
        if path and os.path.exists(path):
            fallback_reported.append(path)

    if fallback_reported:
        return max(fallback_reported, key=os.path.getsize)

    # As a last resort, use the largest playable video file in downloads.
    candidates = []
    for name in os.listdir(output_dir):
        full_path = os.path.join(output_dir, name)
        if not os.path.isfile(full_path):
            continue
        if os.path.splitext(name)[1].lower() not in {".mp4", ".mkv", ".webm", ".mov"}:
            continue
        candidates.append(full_path)

    if candidates:
        return max(candidates, key=os.path.getsize)

    # Bubble up the most helpful error details.
    errors = []
    if primary.returncode != 0 and primary.stderr:
        errors.append(primary.stderr.strip())
    if fallback.returncode != 0 and fallback.stderr:
        errors.append(fallback.stderr.strip())
    detail = "\n\n".join(errors) if errors else "No video file was produced."
    raise RuntimeError(f"yt-dlp failed to produce a playable video file.\n{detail}")


# ─────────────────────────────────────────────
# Quick video metadata (no model needed)
# ─────────────────────────────────────────────
def get_video_info(path: str) -> dict:
    """Return basic metadata for a video file."""
    cap = cv2.VideoCapture(path)
    info = {
        "width"       : int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height"      : int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps"         : round(cap.get(cv2.CAP_PROP_FPS), 2),
        "frame_count" : int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "duration_sec": round(
            cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(cap.get(cv2.CAP_PROP_FPS), 1), 2
        ),
    }
    cap.release()
    return info


# ─────────────────────────────────────────────
# Extract a thumbnail from the video
# ─────────────────────────────────────────────
def extract_thumbnail(path: str, second: float = 1.0) -> np.ndarray | None:
    """
    Return an RGB numpy array of the frame at *second* seconds,
    or None if the video cannot be read.
    """
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(second * fps))
    ret, frame = cap.read()
    cap.release()
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None


# ─────────────────────────────────────────────
# Format seconds → human-readable string
# ─────────────────────────────────────────────
def fmt_time(seconds: float) -> str:
    """Convert seconds to mm:ss string."""
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"
