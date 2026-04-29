# 🚗 SPINNY — Dual-Mode Automotive Shock Absorber Health Detection System

**Developed by Aditya Paswan**

Analyse suspension / shock absorber health from a phone-camera video of a car wheel crossing a speed breaker. The system supports two analysis modes: Rule-Based Marker Mode and Supervised ML AI Mode.

---

## How It Works

1. A phone camera records the **side view of the wheel area** while the car drives over a speed breaker at low speed.
2. The software reads the video frame-by-frame and tracks two reference points using colour markers:
   - **Wheel marker (Bright Yellow)** — placed on the wheel hub / centre cap.
   - **Body marker (Green)** — placed on the fender / body panel directly above the wheel.
3. The vertical gap between these two points changes as the suspension compresses and rebounds over the bump.
4. Signal processing extracts features (compression ratio, rebound, oscillations, frequency) and computes a **Suspension Health Score (0-100%)** with a classification label.

---

## Two Analysis Modes

| Feature | 🟢 Marker Mode (Rule-Based) | 🤖 AI Mode (Supervised ML) |
|---|---|---|
| Scoring | Weighted formula with fixed thresholds | RandomForest + XGBoost + GradientBoosting ensemble |
| Training | No training needed | Requires labelled training videos (Healthy / Too Soft / Too Hard) |
| Features used | 4 metrics (CR, Rebound, Oscillation, Frequency) | 19 features extracted from signal analysis |
| Explainability | Score breakdown shown per metric | SHAP values explain each prediction |
| Best for | Quick analysis with known good thresholds | Higher accuracy when trained on your specific video setup |
| Front + Rear | Single wheel per video | Supports both wheels in a single video |

---

## Marker Mode — Scoring Formula

```
Health = 0.35 × CR_Score + 0.20 × Rebound_Score + 0.25 × Osc_Score + 0.20 × Freq_Score
```

### Classification (Score-Driven — Score and Label Always Agree)

| Condition | Score Range |
|---|---|
| **Healthy** | Score ≥ 65% |
| **Mid-Condition** | Score 43-64% |
| **Faulty** | Score < 43% |

### Sub-Score Computation

**CR Score (0-100):** Based on Compression Ratio (ideal range 0.08-0.30).
- CR in 0.08-0.30 → 100
- CR < 0.08 → linearly scaled 0-100
- CR > 0.30 → linearly drops to 0 at CR=0.45

**Rebound Score (0-100):** Based on Rebound Ratio (rebound height / compression depth).
- Ideal RR 0.5-3.0 → 100
- Special case: if compression < 5px but rebound ≥ 8px → 85 (camera distance issue, suspension is active)
- If CR is extreme (>0.40 or <0.05), rebound score is capped at 40

**Oscillation Score (0-100):** Based on number of bounce cycles after impact.
- 0 cycles → 90 (well-damped)
- 1 cycle → 100 (ideal)
- 2 cycles → 40
- 3 cycles → 5
- 4+ cycles → 0

**Frequency Score (0-100):** Based on dominant oscillation frequency.
- ≤ 3 Hz → 100
- ≥ 8 Hz → 0
- Linear between 3-8 Hz

### Context Penalty

When CR is extreme AND other indicators confirm it:
- CR > 0.45 AND osc ≥ 2 → non-CR scores capped at 30
- CR < 0.05 AND rebound < 8px → non-CR scores capped at 30

This prevents mild-looking sub-scores from masking a clearly faulty suspension.

---

## AI Mode — Supervised ML Classifier

### 19 Features Extracted Per Video

| # | Feature | Description |
|---|---|---|
| 1 | Compression Ratio (CR) | Gap closure during bump (0-1) |
| 2 | Oscillation Count | Number of bounce cycles |
| 3 | Dominant Frequency | Bounce speed (Hz) |
| 4 | Decay Rate | How fast oscillations die out |
| 5 | Settling Time | Time to stabilize (seconds) |
| 6 | Max Velocity | Fastest gap change (px/s) |
| 7 | Max Acceleration | Fastest velocity change (px/s²) |
| 8 | Signal Energy | Total movement energy |
| 9 | Normalized Amplitude | Max displacement relative to baseline |
| 10 | Max Rebound | Rebound height (px) |
| 11 | Max Compression | Compression depth (px) |
| 12 | Compression Depth | Deepest compression point (px) |
| 13 | Zero Crossings | Times signal crossed baseline |
| 14 | Baseline Gap (H0) | Normal gap before bump (px) |
| 15 | Bottoming Out | Wheel hit body? (0/1) |
| 16 | Rebound Ratio | Rebound / Compression (derived) |
| 17 | CR × Oscillation | Interaction feature (derived) |
| 18 | Compression % of H0 | Depth as % of baseline (derived) |
| 19 | Rebound − Compression | Net rebound difference (derived) |

### ML Models (3-Way Soft Voting Ensemble)

1. **RandomForest** — 300 decision trees, class-weight balanced
2. **XGBoost** — 300 gradient-boosted trees, learning rate 0.05
3. **GradientBoosting** — 200 trees, learning rate 0.1

Each model produces class probabilities. The final prediction is the **average of all three models' probabilities** (soft voting). This catches errors that any single model would make.

### Training Pipeline

1. Upload labelled videos (Healthy / Too Soft / Too Hard) — minimum 3 per class
2. For each video: detect markers → compute gap signal → extract 19 features
3. Outlier removal using IsolationForest (removes badly recorded videos)
4. Data augmentation: 3 noise copies per video (2% Gaussian noise on continuous features)
5. StandardScaler normalization
6. Train all 3 models
7. StratifiedKFold cross-validation for accuracy reporting
8. SHAP TreeExplainer for prediction explanations

### Health Percentage (AI Mode)

```
If predicted Healthy:    health_pct = 65 + (p_healthy × 35)     → range 65-100%
If predicted Faulty:     health_pct = 10 + ((1 - p_fault) × 30) → range 10-40%
```

---

## Wheel Center Detection

The system automatically detects the true geometric wheel center using a hybrid approach:

1. **CLAHE** — Contrast enhancement for better edge detection
2. **Gaussian Blur** — Noise reduction
3. **Multi-Scale Hough Circle Transform** — Detects circles at multiple parameter settings
4. **Median Outlier Filtering** — Rejects false circle detections
5. **Geometric Center Averaging** — Computes true center from all valid circles

If the yellow marker is placed off-center on the wheel, the system corrects the offset automatically every frame.

---

## Colour Markers

| Marker | Colour | Placement | HSV Range |
|---|---|---|---|
| **Wheel marker** | Bright Yellow | Wheel hub / centre cap | H: 15-45, S: 100-255, V: 100-255 |
| **Body marker** | Green | Fender / body panel above wheel | H: 35-85, S: 80-255, V: 80-255 |

**Important:** Use **bright yellow** (not red) for the wheel marker. Red was changed to yellow because red is a common car body colour and caused false detections.

Use matte stickers — glossy surfaces cause reflections that break colour detection.

---

## Front + Rear Wheel Analysis

In AI Mode, you can analyse both front and rear wheels from a single video:

1. Record the **full side view** of the car crossing the bump
2. Select "Both (Front + Rear)" in the wheel selection
3. Enter the approximate time (seconds) when the rear wheel hits the bump
4. The software splits the video into two segments and analyses each wheel separately
5. Separate results, graphs, and annotated videos are generated for each wheel

---

## Car Data Entry

The dashboard includes a car information form:
- **Make** (brand: Volkswagen, Maruti, etc.)
- **Model** (Vento, Swift, etc.)
- **Type** (Sedan, Hatchback, SUV, etc.)
- **Registration Number**

Data is stored in `models/car_data.json` with each analysis result. Google Sheets integration is supported with `gspread` library setup.

---

## Installation

```bash
cd suspension_project
pip install -r requirements.txt
pip install scikit-learn xgboost shap joblib    # For AI Mode
```

### Requirements

- Python 3.8+
- OpenCV (`opencv-python`)
- NumPy, Pandas, Matplotlib
- Streamlit (for dashboard)
- scikit-learn, XGBoost, SHAP, joblib (for AI Mode)

---

## Folder Structure

```
suspension_project/
├── app.py                           # Streamlit dashboard (Marker + AI Mode)
├── main.py                          # CLI entry point
├── config.yaml                      # All tunable parameters
├── train_model.py                   # CLI model training script
├── evaluate_model.py                # CLI model evaluation script
├── generate_test_videos.py          # Synthetic test video generator
├── debug_scores.py                  # Score verification tool
├── requirements.txt
├── README.md
├── src/
│   ├── __init__.py
│   ├── video_loader.py              # Video I/O and metadata
│   ├── detector_marker.py           # HSV colour marker detection (yellow + green)
│   ├── detector_ai.py               # YOLO + classical CV detection
│   ├── tracker.py                   # OpenCV trackers + Kalman
│   ├── signal_analysis.py           # Smoothing, peak detection, 19 features
│   ├── health_scoring.py            # Rule-based scoring (4-metric formula)
│   ├── ml_classifier.py             # Supervised ML classifier (RF + XGB + GB)
│   ├── training_data_manager.py     # Training video registry manager
│   ├── visualization.py             # Matplotlib plots (4 graphs)
│   ├── exporter.py                  # CSV + JSON export
│   ├── wheel_center.py              # Hough circle wheel center detection
│   ├── pipeline.py                  # Main orchestration (marker + ML modes)
│   └── utils.py                     # Shared helpers
├── models/
│   ├── suspension_ml_model.pkl      # Trained ML model (created at training time)
│   ├── training_registry.json       # Training video registry
│   └── car_data.json                # Car information records
├── data/
│   ├── input_videos/                # Videos to analyse
│   ├── training_videos/
│   │   ├── Healthy/
│   │   ├── Faulty_Too_Soft/
│   │   └── Faulty_Too_Hard/
│   └── output/                      # Analysis results
```

---

## Usage

### Streamlit Dashboard (Recommended)

```bash
python -m streamlit run app.py
```

### CLI — Marker Mode

```bash
python main.py --video data/input_videos/test.mp4 --mode marker
```

### CLI — Train ML Model

```bash
python train_model.py --healthy data/training_videos/Healthy/ --too-soft data/training_videos/Faulty_Too_Soft/ --too-hard data/training_videos/Faulty_Too_Hard/
```

### CLI — Evaluate ML Model

```bash
python evaluate_model.py --test-dir data/test_videos/ --model models/suspension_ml_model.pkl
```

---

## Output Files

| File | Description |
|---|---|
| `00_health_report.png` | Visual health report card with score breakdown |
| `01_raw_gap.png` | Raw vertical gap over time |
| `02_smoothed_gap.png` | Smoothed gap with compression and rebound markers |
| `03_displacement.png` | Displacement from baseline with peaks, valleys, and CR/Osc/RR info |
| `04_marker_tracking.png` | Yellow and green dot positions on white background |
| `annotated.mp4` | Video with tracked points and gap overlay (always saved) |
| `frame_data.csv` | Per-frame tracking data |
| `summary.json` | Full analysis summary |

---

## Video Recording Guidelines

### Camera Setup

- Resolution: **720p (1280×720) or higher** — higher resolution = more accurate
- Frame rate: **60 FPS** recommended (30 FPS minimum)
- Mount on a **tripod** — camera shake ruins the measurement
- Position at **90° side-on** to the wheel, at **wheel-centre height**
- Distance: **1.5 to 2.5 meters** from the wheel — the wheel should fill at least 1/4 of the frame
- **Keep the same distance and resolution** for all videos (training and analysis)
- No zoom changes during recording
- Good, even lighting — avoid direct sun glare

### Marker Placement

- **Wheel marker (Bright Yellow)** — one matte circular sticker (~3-5 cm) on the wheel hub / centre cap
- **Body marker (Green)** — one matte circular sticker on the fender directly above the wheel
- Do NOT place markers on the rotating tyre tread
- Ensure marker colours contrast with car body colour
- Use matte stickers — glossy surfaces cause specular reflections

### For Front + Rear Analysis

- Record the **full side** of the car from entry to exit
- Place markers on BOTH front and rear wheels (yellow on hub, green above each wheel)
- The car should pass through the bump at constant low speed (~10-15 km/h)

---

## Troubleshooting

| Problem | Solution |
|---|---|
| Marker not detected | Check if yellow/green stickers are bright enough. Adjust HSV ranges in `config.yaml` |
| Hmin = 0.0px | Replace `signal_analysis.py` with latest version — outlier filter fix |
| CR = 1.0 (100%) | Replace `pipeline.py` — empty frame stripping fix |
| Score disagrees with label | Replace `health_scoring.py` — unified score-driven classification |
| Video too dark | Re-record with better lighting |
| ML accuracy low | Add more training videos (10+ per class recommended) |
| Annotated video not saving | Replace `pipeline.py` — always-save fix |
| Wrong colour detected | Marker colour is now bright yellow (not red). Update stickers. |

---

## Key Bug Fixes Applied

1. **Hmin = 0 bug** — Fixed by computing min/max from valid detections only (not interpolated values), with outlier filtering (4σ + 50px tolerance)
2. **CR > 1.0 bug** — Fixed by capping CR at 0.95 and depth at 95% of baseline
3. **Empty frames bug** — Fixed by stripping leading/trailing frames with no marker detections before analysis
4. **Score vs Label disagreement** — Fixed by making label derive FROM score (unified system)
5. **Baseline from garbage** — Fixed by sliding window to find most stable region for baseline computation
6. **Red marker interference** — Changed wheel marker from red to bright yellow

---

## Technology Stack

- **Python 3.8+** — Core language
- **OpenCV** — Video processing, marker detection, Hough circles
- **NumPy / SciPy** — Signal processing, Savitzky-Golay smoothing
- **Matplotlib** — Graph generation
- **Streamlit** — Web dashboard
- **scikit-learn** — RandomForest, GradientBoosting, StandardScaler, IsolationForest
- **XGBoost** — Gradient boosted trees
- **SHAP** — Model explanation (TreeExplainer)
- **joblib** — Model serialization

---

## Author

**Aditya Paswan**

SPINNY — Suspension Health Detection System
