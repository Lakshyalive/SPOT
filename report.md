# Technical Report
## Multi-Object Detection and Persistent ID Tracking in Sports Footage

---

### 1. Detector: YOLOv8

**Model used:** YOLOv8 (nano / small / medium — user-selectable)

YOLOv8 by Ultralytics is a real-time single-stage object detector that predicts bounding boxes
and class probabilities in a single forward pass. It was chosen because:

- **Speed**: The nano variant runs at 30–60 FPS on a modern CPU — acceptable for offline processing.
- **Accuracy**: Trained on COCO 80 classes, including `person` (class 0), which covers all athletes/participants.
- **Ecosystem**: The `ultralytics` library provides a single `model.track()` API that internally calls the tracker, removing the need to wire them together manually.

The detector outputs, per frame:
- Bounding box coordinates `(x1, y1, x2, y2)`
- Class ID and human-readable label
- Confidence score

---

### 2. Tracker: ByteTrack

**Algorithm:** ByteTrack (integrated into ultralytics via `bytetrack.yaml`)

ByteTrack is a multi-object tracking algorithm that improves on classic SORT-style trackers by **keeping low-confidence detections** in the matching pipeline instead of discarding them.

**How ID consistency is maintained:**

1. **Kalman filter prediction** — Each active tracklet predicts its next position based on velocity.
2. **Two-round Hungarian matching:**
   - Round 1: Match high-confidence detections to existing tracks using IoU.
   - Round 2: Match remaining tracks against low-confidence detections — this recovers objects that are partially occluded or motion-blurred.
3. **Track lifecycle** — New tracks are created for unmatched detections; lost tracks are kept in a "lost" buffer for a few frames before being deleted.

The `persist=True` flag in `model.track()` tells ultralytics to maintain track state across successive calls (i.e., across frames).

---

### 3. Why This Combination?

| Criterion | YOLOv8 + ByteTrack |
|-----------|-------------------|
| Detection quality | High — COCO-pretrained, person class well-covered |
| Tracking robustness | High — two-round matching handles occlusion |
| Integration effort | Low — single API call handles both |
| Speed | Configurable via model size + frame skip |
| Open source | Yes — no paid API or dataset required |

---

### 4. Challenges Faced

**Occlusion:** When two players closely overlap, the Kalman prediction can drift and cause an ID switch on separation. ByteTrack mitigates this with its low-confidence matching round but cannot fully eliminate it.

**Similar appearance:** Players in the same team uniform look very similar. Since ByteTrack uses only spatial/IoU information (not appearance), IDs can swap when players cross paths.

**Camera motion:** Rapid pans or zooms shift all bounding boxes simultaneously, which can confuse the IoU-based matcher. Frame skip partially helps by giving each step more motion context.

**Confidence threshold tuning:** Too low → false positive detections of crowd members in background. Too high → players in partial occlusion are missed, breaking track continuity.

---

### 5. Failure Cases Observed

- Long occlusion (> 5 frames) frequently causes ID resets for the occluded person.
- Tracking players that exit and re-enter the frame assigns them new IDs (no re-ID module).
- Very small/distant players (e.g., crowd members) get detected if confidence is set too low.

---

### 6. Possible Improvements

| Improvement | Description |
|-------------|-------------|
| Re-ID module | Add appearance embedding (e.g., OSNet) to recover IDs after long occlusion |
| StrongSORT / BoT-SORT | Alternative trackers with built-in appearance features |
| Larger YOLO model | `yolov8l` / `yolov8x` for more accurate detections |
| Sport-specific fine-tuning | Fine-tune on cricket/football dataset for better player detection |
| Team clustering | K-means on jersey color histograms to group players by team |
| Speed estimation | Use camera homography + known pitch dimensions for real-world speed |
| Trajectory heatmap | Accumulate centre points on a pitch diagram for tactical analysis |

---

### 7. Pipeline Summary

```
Input Video
    │
    ├─ Frame extracted by OpenCV
    │
    ▼
YOLOv8 Detector
    ├─ Bounding boxes
    ├─ Class labels
    └─ Confidence scores
    │
    ▼
ByteTrack
    ├─ Kalman filter prediction
    ├─ Hungarian matching (2 rounds)
    └─ Persistent track IDs
    │
    ▼
Annotated Frame
    ├─ Bounding box (unique color per ID)
    ├─ ID badge + label + confidence
    └─ Movement trail (optional)
    │
    ▼
Output MP4 Video
```

**Total pipeline latency** (yolov8n, CPU, 720p):
- ~30–60 ms per frame → ~15–30 processed FPS
- Use `frame_skip=2` or `frame_skip=3` to reduce processing time on long videos.
