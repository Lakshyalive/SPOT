"""
tracker.py
----------
Core detection and tracking pipeline.
Uses YOLOv8 for detection and ByteTrack (built into ultralytics) for tracking.
"""

import cv2
import numpy as np
from collections import defaultdict
import time


# ─────────────────────────────────────────────
# Color palette — one unique color per track ID
# ─────────────────────────────────────────────
def get_color(track_id: int) -> tuple:
    """Return a consistent BGR color for a given track ID."""
    np.random.seed(track_id * 7 + 13)          # deterministic per ID
    return tuple(int(c) for c in np.random.randint(80, 255, size=3))


# ─────────────────────────────────────────────
# Draw one bounding box + label on a frame
# ─────────────────────────────────────────────
def draw_box(frame: np.ndarray, x1, y1, x2, y2,
             track_id: int, label: str, conf: float) -> np.ndarray:
    """
    Draw a rounded bounding box with ID badge on the frame.

    Parameters
    ----------
    frame     : BGR image (numpy array)
    x1,y1     : top-left corner of the box
    x2,y2     : bottom-right corner of the box
    track_id  : unique integer ID assigned by the tracker
    label     : YOLO class label (e.g. 'person')
    conf      : detection confidence (0.0 – 1.0)

    Returns
    -------
    frame with annotation drawn on it (in-place)
    """
    color = get_color(track_id)

    # Main bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=2)

    # Label text
    text = f"ID:{track_id} {label} {conf:.0%}"
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                          0.55, 1)

    # Filled badge behind text so it's readable
    badge_y1 = max(y1 - th - baseline - 6, 0)
    badge_y2 = y1
    cv2.rectangle(frame, (x1, badge_y1), (x1 + tw + 6, badge_y2), color, -1)

    cv2.putText(frame, text,
                (x1 + 3, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)

    return frame


# ─────────────────────────────────────────────
# Optional: draw trajectory tail for each ID
# ─────────────────────────────────────────────
def draw_trail(frame: np.ndarray,
               history: dict,
               track_id: int,
               max_points: int = 30) -> np.ndarray:
    """
    Draw a fading polyline trail for a tracked subject.

    Parameters
    ----------
    frame      : BGR image
    history    : dict mapping track_id -> list of (cx, cy) centre points
    track_id   : ID whose trail to draw
    max_points : how many past positions to include in the trail
    """
    pts = history.get(track_id, [])[-max_points:]
    color = get_color(track_id)

    for i in range(1, len(pts)):
        alpha = i / len(pts)                    # fade older points
        faded = tuple(int(c * alpha) for c in color)
        thickness = max(1, int(3 * alpha))
        cv2.line(frame, pts[i - 1], pts[i], faded, thickness)

    return frame


# ─────────────────────────────────────────────
# Main processing function
# ─────────────────────────────────────────────
def process_video(
    input_path: str,
    output_path: str,
    model_size: str = "yolov8n.pt",
    confidence: float = 0.35,
    classes: list = None,           # None → detect everything; [0] → only persons
    show_trails: bool = True,
    frame_skip: int = 1,            # process every Nth frame (1 = every frame)
    progress_callback=None,         # optional callable(current, total)
) -> dict:
    """
    Run detection + tracking on *input_path* and write annotated video to
    *output_path*.

    Returns a summary dict with statistics.
    """

    # Import YOLO lazily so Streamlit can start quickly on constrained cloud runtimes.
    from ultralytics import YOLO

    # ── Load model ──────────────────────────────────────────────────────────
    model = YOLO(model_size)          # downloads automatically on first run

    # ── Open video ──────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # ── Video writer ─────────────────────────────────────────────────────────
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(output_path, fourcc, fps / max(frame_skip, 1),
                              (width, height))

    # ── State ────────────────────────────────────────────────────────────────
    trail_history   = defaultdict(list)   # track_id → list of centre points
    all_ids_seen    = set()
    frame_idx       = 0
    start_time      = time.time()
    last_frame      = None                # keep last annotated frame for display

    # ── Process frames ───────────────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # Skip frames for speed (but still write them so video length stays correct)
        if frame_idx % frame_skip != 0:
            out.write(frame)
            continue

        # ── Run YOLO + ByteTrack ─────────────────────────────────────────────
        results = model.track(
            source=frame,
            persist=True,           # keeps track IDs consistent across calls
            conf=confidence,
            classes=classes,
            tracker="bytetrack.yaml",
            verbose=False,
        )

        annotated = frame.copy()

        if results and results[0].boxes is not None:
            boxes = results[0].boxes

            # Each box may or may not have a track ID yet
            ids     = boxes.id.int().cpu().numpy()   if boxes.id   is not None else []
            xyxys   = boxes.xyxy.int().cpu().numpy()
            confs   = boxes.conf.float().cpu().numpy()
            cls_ids = boxes.cls.int().cpu().numpy()

            for i, (xyxy, conf_val, cls_id) in enumerate(zip(xyxys, confs, cls_ids)):
                x1, y1, x2, y2 = xyxy
                label = model.names[int(cls_id)]

                track_id = int(ids[i]) if len(ids) > i else -1
                all_ids_seen.add(track_id)

                # Centre point for trail
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                if track_id >= 0:
                    trail_history[track_id].append((cx, cy))

                # Draw trail first (so box appears on top)
                if show_trails and track_id >= 0:
                    annotated = draw_trail(annotated, trail_history, track_id)

                # Draw bounding box
                annotated = draw_box(annotated, x1, y1, x2, y2,
                                     track_id, label, conf_val)

        # ── Overlay frame counter ────────────────────────────────────────────
        cv2.putText(annotated,
                    f"Frame {frame_idx}/{total}  |  IDs seen: {len(all_ids_seen)}",
                    (10, height - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        out.write(annotated)
        last_frame = annotated

        # ── Report progress ──────────────────────────────────────────────────
        if progress_callback:
            progress_callback(frame_idx, total)

    cap.release()
    out.release()

    elapsed = time.time() - start_time

    return {
        "total_frames"   : frame_idx,
        "unique_ids"     : len(all_ids_seen),
        "elapsed_sec"    : round(elapsed, 1),
        "fps_processed"  : round(frame_idx / elapsed, 1),
        "last_frame"     : last_frame,
    }
