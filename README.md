# 🏃 SPOT — Sports Player Object Tracking

**Sports Player Object Tracking in Public Sports / Event Footage**

Uses **YOLOv8** for detection and **ByteTrack** for persistent ID tracking,
wrapped in a clean **Streamlit** web UI.

Live app: https://spotapp.streamlit.app/

## Demo Video
Watch here: [SPOT Demo Video](https://drive.google.com/file/d/1iiHAtASq8IUMNXDqkXXBbg3SLNSTafaq/view?usp=sharing)

---

## Project Structure

```
spot/
├── app.py            ← Streamlit front-end (run this)
├── tracker.py        ← Core detection + tracking pipeline
├── utils.py          ← Video download, metadata helpers
├── requirements.txt
├── README.md
└── report.md         ← Short technical report
```

---

## Installation

### 1. Clone / unzip this project
```bash
cd spot
```

### 2. (Recommended) Create a virtual environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

> **GPU note**: If you have an NVIDIA GPU, install the CUDA version of PyTorch first
> for much faster processing:
> https://pytorch.org/get-started/locally/

### 4. (Optional) Install yt-dlp for YouTube downloads
```bash
pip install yt-dlp
```

---

## Running the App

```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501`

---

## How to Use

1. **Upload a video** (MP4 / AVI / MOV) **or paste a YouTube URL**.
   - Note: YouTube URL download support is currently under work.
2. Adjust settings in the left sidebar:
   - **Model** – `yolov8n` is fastest; `yolov8m` is most accurate
   - **Confidence** – lower detects more, higher detects fewer (but surer)
   - **Frame Skip** – `1` = every frame (best quality); `3` = 3× faster
   - **Trails** – toggle movement tail overlay
3. Click **▶ START TRACKING**.
4. Watch the progress bar, then **download** your annotated video.

---

## Assumptions

- Input video is MP4 or compatible format readable by OpenCV.
- People/athletes are the primary tracking targets (class 0 in COCO).
- The pipeline is CPU-compatible but runs faster on GPU.
- ByteTrack parameters use ultralytics defaults (tuned for general tracking).

---

## Limitations

- ID switches can occur during prolonged overlaps between subjects.
- Very fast motion or motion blur may cause missed detections.
- Works best on 480p–720p video.
- Does not perform team classification or re-identification across full video cuts.
- YouTube URL download feature is under active development and may fail for some links.

---

## Model / Tracker Choices

| Component | Choice | Reason |
|-----------|--------|--------|
| Detector | YOLOv8 (nano/small/medium) | Best accuracy/speed tradeoff; integrated tracker support |
| Tracker | ByteTrack | Handles low-confidence detections well; robust to partial occlusion |
| Framework | Ultralytics | Single `model.track()` call handles both detection + tracking |

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `ultralytics` | YOLOv8 + ByteTrack |
| `opencv-python` | Video read/write, drawing |
| `streamlit` | Web UI |
| `yt-dlp` | YouTube download (optional) |
| `torch` | Deep learning runtime |
