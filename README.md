# Suspension-Health-Detection-system
# Dual-Mode Automotive Shock Absorber Health Detection System

Analyse suspension / shock absorber health from a phone-camera video of a wheel
crossing a speed breaker.

---

## How It Works

1. A phone camera records the **side view of one wheel area** while the car
   drives over a speed breaker at low speed.
2. The software reads the video frame-by-frame and tracks two reference points:
   - **Wheel reference** — the wheel center / hub area.
   - **Body reference** — the wheel-well / fender edge above the wheel.
3. The vertical gap between these points changes as the suspension compresses
   and rebounds.
4. Signal processing extracts oscillation count, settling time, amplitude,
   and computes a **Suspension Health Percentage (0-100)** with a label:
   **Good**, **Mid-condition**, or **Faulty**.

---

## Two Modes

| Feature | AI Mode (`--mode ai`) | Marker Mode (`--mode marker`) |
|---|---|---|
| Detection | YOLO + classical CV fallback | Colour markers on vehicle |
| Setup | No physical markers needed | Stick two coloured circles on car |
| Reliability | Depends on scene; may need manual fallback | Very reliable with good markers |
| Best for | Quick field test | First working prototype / repeatable tests |

### Why Marker Mode Is the Recommended First Prototype

Marker-based detection is **far more reliable** under varying lighting and
camera conditions.  Colour segmentation + centroid tracking provides sub-pixel
stability that is hard to match with generic object detectors on an unseen
vehicle.

### Why the Wheel Marker Goes Near the Hub Centre

The outer tyre surface **rotates**.  Placing a marker on the tread would cause
it to orbit the wheel, making vertical tracking meaningless.  The hub / wheel
centre area moves **only vertically** (with minor lateral motion) as the
suspension compresses and rebounds — exactly the signal we need.

### Why Markers Are Circular

A circular marker has a **rotation-invariant centroid**.  Regardless of slight
camera angle changes or partial occlusion, the centroid of a detected circle
is the most stable 2-D point estimator.

---

## Installation

```bash
cd suspension_dual_mode_project
pip install -r requirements.txt
```

If you want AI mode with YOLO, download weights:

```bash
# Example: YOLOv8 nano
pip install ultralytics
yolo export model=yolov8n.pt   # or just place yolov8n.pt in models/
```

---

## Folder Structure

```
suspension_dual_mode_project/
├── main.py                  # CLI entry point
├── config.yaml              # All tunable parameters
├── requirements.txt
├── README.md
├── src/
│   ├── __init__.py
│   ├── video_loader.py      # Video I/O and metadata
│   ├── detector_ai.py       # YOLO + classical CV detection
│   ├── detector_marker.py   # HSV colour marker detection
│   ├── tracker.py           # OpenCV trackers + Kalman
│   ├── signal_analysis.py   # Smoothing, peak detection, features
│   ├── health_scoring.py    # Rule-based scoring 0-100
│   ├── visualization.py     # Matplotlib plots
│   ├── exporter.py          # CSV + JSON export
│   ├── utils.py             # Shared helpers
│   └── pipeline.py          # Main orchestration
├── models/                  # YOLO weights (optional)
├── data/
│   ├── input_videos/
│   ├── reference_videos/
│   └── output/              # Results go here
└── notebooks/
```

---

## Usage

### Basic Marker Mode

```bash
python main.py --video data/input_videos/test.mp4 --mode marker
```

### AI Mode

```bash
python main.py --video data/input_videos/test.mp4 --mode ai
```

### With Manual Fallback

```bash
python main.py --video data/input_videos/test.mp4 --mode ai --manual-fallback
```

### Save Annotated Video

```bash
python main.py --video data/input_videos/test.mp4 --mode marker --save-video
```

### With Wheel Diameter Calibration

```bash
python main.py --video data/input_videos/test.mp4 --mode marker --wheel-diameter-mm 600
```

### Compare Against Reference

```bash
python main.py --video data/input_videos/test.mp4 --mode marker \
               --reference-video data/reference_videos/healthy.mp4
```

---

## Video Capture Guidelines

### Camera Setup

- Use a **phone camera at 60 FPS or higher**.
- Mount on a **tripod** — stability is critical.
- Position at **90° side-on** to the wheel, at roughly **wheel-centre height**.
- Frame the shot so the **wheel and wheel arch are fully visible**.
- Good, even lighting.  Avoid direct sun glare on the wheel.
- **No zoom changes** during recording.
- Keep the car's tested wheel **fully in frame** throughout the bump crossing.

### Marker Placement (for `--mode marker`)

- **Wheel marker** — one circular sticker (~3-5 cm) near the wheel hub /
  centre cap.  Use a **red** sticker (default config).
- **Body marker** — one circular sticker on the body panel / fender directly
  above the wheel.  Use a **green** sticker (default config).
- Use **matte** stickers — glossy / reflective surfaces cause specular
  highlights that break colour detection.
- Ensure the marker colours contrast strongly with the car paint.
- Do **NOT** place the wheel marker on the rotating tyre tread.

---

## Outputs

| File | Description |
|---|---|
| `frame_data.csv` | Per-frame tracking and gap data |
| `summary.json` | Oscillations, settling time, health score, label |
| `01_raw_gap.png` | Raw vertical gap over time |
| `02_smoothed_gap.png` | Smoothed gap with baseline |
| `03_peaks_valleys.png` | Detected oscillation peaks and valleys |
| `04_settling.png` | Settling time with tolerance band |
| `05_comparison.png` | Overlay of reference vs test (if provided) |
| `annotated.mp4` | Video with tracked points drawn (if `--save-video`) |

---

## Troubleshooting

| Problem | Solution |
|---|---|
| Marker not detected | Adjust HSV ranges in `config.yaml`.  Print a frame, check HSV values with an online picker. |
| Too many oscillations | Increase `peak_prominence_ratio` or `peak_distance_frames` in config to reject jitter. |
| AI mode fails | Ensure YOLO weights exist at the configured path, or use `--manual-fallback`. |
| Video too dark | Re-record with better lighting or adjust `marker_detection.blur_kernel_size`. |
| Score seems wrong | Tune `health_scoring` thresholds for your vehicle class. |

---

## Future Improvements

- Train a custom YOLO model on wheel + wheel-arch data for robust AI mode.
- Add stereo camera support for absolute distance measurement.
- Implement frequency-domain analysis (FFT) for damping ratio estimation.
- Mobile app with real-time overlay.
- Multi-wheel simultaneous analysis.
- Database of known-good signatures per vehicle model.
