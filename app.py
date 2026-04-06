"""
app.py
------
Streamlit front-end for SPOT — Sports Player Object Tracking.

Run with:
    streamlit run app.py
"""

import streamlit as st
import os
import tempfile
import time
import cv2

from tracker import process_video
from utils   import download_video, get_video_info, extract_thumbnail, fmt_time


# ════════════════════════════════════════════════════════════════════
#  Page config  (must be the FIRST streamlit call)
# ════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title  = "SPOT — Sports Player Object Tracking",
    page_icon   = "🏃",
    layout      = "wide",
)


# ════════════════════════════════════════════════════════════════════
#  Custom CSS  – clean dark sports UI
# ════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Global ── */
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=Inter:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0d0d0f;
    color: #e8e8ec;
}

/* ── Main title ── */
.hero {
    text-align: center;
    padding: 2rem 0 1rem;
}
.hero h1 {
    font-family: 'Rajdhani', sans-serif;
    font-size: 3rem;
    font-weight: 700;
    letter-spacing: 3px;
    background: linear-gradient(135deg, #f0a500 0%, #e05c00 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.hero p {
    color: #888;
    font-size: 0.95rem;
    margin-top: 0.3rem;
    letter-spacing: 1px;
    text-transform: uppercase;
}

/* ── Section headings ── */
.section-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.25rem;
    font-weight: 600;
    color: #f0a500;
    letter-spacing: 2px;
    text-transform: uppercase;
    border-left: 3px solid #f0a500;
    padding-left: 0.6rem;
    margin-bottom: 0.8rem;
}

/* ── Stat cards ── */
.stat-row {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
    margin: 1rem 0;
}
.stat-card {
    flex: 1;
    min-width: 130px;
    background: #1a1a1f;
    border: 1px solid #2a2a32;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.stat-card .val {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #f0a500;
    line-height: 1;
}
.stat-card .lbl {
    font-size: 0.72rem;
    color: #666;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 0.25rem;
}

/* ── Info box ── */
.info-box {
    background: #12121a;
    border: 1px solid #2a2a32;
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    font-size: 0.88rem;
    color: #aaa;
    line-height: 1.7;
}

/* ── Streamlit overrides ── */
[data-testid="stSidebar"] {
    background: #10101a;
    border-right: 1px solid #1e1e28;
}
[data-testid="stSidebar"] label {
    color: #ccc !important;
    font-size: 0.85rem;
}
.stButton > button {
    background: linear-gradient(135deg, #f0a500, #e05c00) !important;
    color: white !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    letter-spacing: 2px !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 2rem !important;
    width: 100%;
}
.stButton > button:hover {
    opacity: 0.9;
}
div[data-testid="stProgress"] > div {
    background: linear-gradient(90deg, #f0a500, #e05c00) !important;
    border-radius: 4px !important;
}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
#  Hero header
# ════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
    <h1>🏃 SPOT</h1>
    <p>Sports Player Object Tracking · YOLOv8 + ByteTrack</p>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
#  Sidebar  — settings
# ════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="section-title">⚙ Settings</div>', unsafe_allow_html=True)

    # ── Model size ───────────────────────────────────────────────────
    model_size = st.selectbox(
        "YOLOv8 Model",
        ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
        index=0,
        help=(
            "n = nano (fastest, less accurate)\n"
            "s = small (balanced)\n"
            "m = medium (more accurate, slower)"
        ),
    )

    # ── Confidence ───────────────────────────────────────────────────
    confidence = st.slider(
        "Detection Confidence",
        min_value=0.10, max_value=0.90,
        value=0.35, step=0.05,
        help="Lower → detect more objects (more false positives). Higher → fewer but surer detections.",
    )

    # ── Classes ──────────────────────────────────────────────────────
    detect_class = st.radio(
        "Detect",
        ["People only (faster)", "All objects"],
        index=0,
    )
    classes = [0] if detect_class == "People only (faster)" else None

    # ── Frame skip ───────────────────────────────────────────────────
    frame_skip = st.select_slider(
        "Frame Skip",
        options=[1, 2, 3, 5],
        value=1,
        help="Process every Nth frame. 1 = every frame (slowest, best quality). 3 = 3× faster.",
    )

    # ── Trails ───────────────────────────────────────────────────────
    show_trails = st.checkbox("Show movement trails", value=True,
                               help="Draw a fading tail behind each tracked subject.")

    st.divider()
    st.markdown("""
    <div class="info-box">
        <b>Pipeline</b><br>
        YOLOv8 → ByteTrack<br><br>
        <b>What each setting does</b><br>
        • <i>Model</i>: accuracy vs speed<br>
        • <i>Confidence</i>: detection threshold<br>
        • <i>Frame Skip</i>: speed vs quality<br>
        • <i>Trails</i>: trajectory overlay
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
#  Main area  — two columns
# ════════════════════════════════════════════════════════════════════
col_left, col_right = st.columns([1, 1], gap="large")

# ── Left column: video input ─────────────────────────────────────────
with col_left:
    st.markdown('<div class="section-title">📥 Video Input</div>',
                unsafe_allow_html=True)

    input_method = st.radio("Source", ["Upload a video file", "YouTube / public URL"],
                             horizontal=True, label_visibility="collapsed")

    input_video_path = None

    if input_method == "Upload a video file":
        uploaded = st.file_uploader(
            "Drop your video here",
            type=["mp4", "avi", "mov", "mkv"],
            label_visibility="collapsed",
        )
        if uploaded:
            # Save to a temp file so OpenCV can read it
            suffix = os.path.splitext(uploaded.name)[-1]
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(uploaded.read())
            tmp.flush()
            input_video_path = tmp.name

    else:  # URL
        url = st.text_input(
            "Paste a YouTube or public video URL",
            placeholder="https://www.youtube.com/watch?v=...",
            label_visibility="collapsed",
        )
        if url:
            if st.button("⬇ Download Video", key="dl"):
                with st.spinner("Downloading via yt-dlp …"):
                    try:
                        input_video_path = download_video(url, output_dir="downloads")
                        st.session_state["downloaded_path"] = input_video_path
                        st.success(f"Downloaded: {os.path.basename(input_video_path)}")
                    except Exception as e:
                        st.error(f"Download failed: {e}")
            # Persist across re-runs
            if "downloaded_path" in st.session_state:
                cached_path = st.session_state["downloaded_path"]
                if os.path.exists(cached_path):
                    input_video_path = cached_path
                else:
                    st.session_state.pop("downloaded_path", None)
                    st.warning("Previously downloaded file no longer exists. Please download again.")

    # ── Show video info once we have a path ──────────────────────────
    if input_video_path and os.path.exists(input_video_path):
        info = get_video_info(input_video_path)

        thumb = extract_thumbnail(input_video_path)
        if thumb is not None:
            st.image(thumb, caption="Preview frame", use_container_width=True)

        st.markdown(f"""
        <div class="stat-row">
            <div class="stat-card">
                <div class="val">{info['width']}×{info['height']}</div>
                <div class="lbl">Resolution</div>
            </div>
            <div class="stat-card">
                <div class="val">{info['fps']}</div>
                <div class="lbl">FPS</div>
            </div>
            <div class="stat-card">
                <div class="val">{fmt_time(info['duration_sec'])}</div>
                <div class="lbl">Duration</div>
            </div>
            <div class="stat-card">
                <div class="val">{info['frame_count']}</div>
                <div class="lbl">Frames</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ── Right column: run + results ──────────────────────────────────────
with col_right:
    st.markdown('<div class="section-title">🚀 Run Tracker</div>',
                unsafe_allow_html=True)

    run_btn = st.button(
        "▶ START TRACKING",
        disabled=(input_video_path is None or not os.path.exists(input_video_path)),
    )

    # Status placeholders
    status_text  = st.empty()
    progress_bar = st.empty()
    result_area  = st.empty()

    if run_btn and input_video_path:

        # Output file (temp MP4)
        out_fd, output_path = tempfile.mkstemp(suffix="_tracked.mp4")
        os.close(out_fd)

        # ── Progress callback ─────────────────────────────────────────
        prog_holder = st.empty()
        bar_holder  = st.progress(0)
        start_time  = time.time()

        def update_progress(current: int, total: int):
            pct = current / max(total, 1)
            elapsed = time.time() - start_time
            eta_sec = (elapsed / max(current, 1)) * (total - current)
            bar_holder.progress(pct)
            prog_holder.markdown(
                f"⏳ Processing frame **{current}** / {total} "
                f"— ETA **{fmt_time(eta_sec)}**"
            )

        # ── Run ───────────────────────────────────────────────────────
        with st.spinner("Loading YOLO model and processing …"):
            try:
                stats = process_video(
                    input_path        = input_video_path,
                    output_path       = output_path,
                    model_size        = model_size,
                    confidence        = confidence,
                    classes           = classes,
                    show_trails       = show_trails,
                    frame_skip        = frame_skip,
                    progress_callback = update_progress,
                )
                bar_holder.progress(1.0)
                prog_holder.empty()

                # ── Success ───────────────────────────────────────────
                st.success("✅ Tracking complete!")

                # Stats
                st.markdown(f"""
                <div class="stat-row">
                    <div class="stat-card">
                        <div class="val">{stats['unique_ids']}</div>
                        <div class="lbl">Unique IDs</div>
                    </div>
                    <div class="stat-card">
                        <div class="val">{stats['total_frames']}</div>
                        <div class="lbl">Frames Processed</div>
                    </div>
                    <div class="stat-card">
                        <div class="val">{stats['fps_processed']}</div>
                        <div class="lbl">FPS Achieved</div>
                    </div>
                    <div class="stat-card">
                        <div class="val">{fmt_time(stats['elapsed_sec'])}</div>
                        <div class="lbl">Total Time</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Last annotated frame preview
                if stats.get("last_frame") is not None:
                    import cv2
                    preview_rgb = cv2.cvtColor(stats["last_frame"], cv2.COLOR_BGR2RGB)
                    st.image(preview_rgb, caption="Last processed frame",
                             use_container_width=True)

                # ── Download button ───────────────────────────────────
                with open(output_path, "rb") as f:
                    st.download_button(
                        label     = "⬇ Download Annotated Video",
                        data      = f,
                        file_name = "tracked_output.mp4",
                        mime      = "video/mp4",
                    )

            except Exception as e:
                bar_holder.empty()
                prog_holder.empty()
                st.error(f"❌ Error: {e}")
                st.exception(e)


# ════════════════════════════════════════════════════════════════════
#  How-it-works expandable section at the bottom
# ════════════════════════════════════════════════════════════════════
st.divider()
with st.expander("📖 How does this work?"):
    st.markdown("""
### Pipeline Overview

```
Video Frame
    │
    ▼
┌────────────────────┐
│   YOLOv8 Detector  │  ← detects people / objects, returns bounding boxes + confidence
└────────────────────┘
    │
    ▼
┌────────────────────┐
│   ByteTrack        │  ← assigns persistent IDs by matching detections across frames
└────────────────────┘
    │
    ▼
Annotated Frame (box + ID + trail)
    │
    ▼
Output Video (.mp4)
```

### Model Choices
| Component | Choice | Why |
|-----------|--------|-----|
| Detector | YOLOv8 (nano/small/medium) | Fast, accurate, easy to use |
| Tracker | ByteTrack | Robust to occlusion, uses both high- and low-confidence detections |

### ID Consistency
ByteTrack uses a **Kalman filter** to predict where each object will be next frame, and
**Hungarian algorithm** to match predictions to new detections — giving stable IDs even
when objects overlap or leave the frame briefly.

### Limitations
- IDs may switch if two people overlap for many frames
- Very fast motion can cause missed detections
- Works best with 720p or lower video
    """)
