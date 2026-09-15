# Vision-Based Autonomous Lane and Obstacle Perception Pipeline

An integrated, CPU-optimized computer vision system for autonomous vehicle navigation and Advanced Driver Assistance Systems (ADAS). The pipeline fuses classical geometric computer vision (color-space thresholding, directional Sobel filtering, Inverse Perspective Mapping homography, and 2nd-degree polynomial curve fitting) with deep learning object detection to concurrently compute road curvature, vehicle lateral deviation, and collision proximity risk in real-time.

**Course:** Computer Vision (CSE3010) — VIT Bhopal University  
**Author:** Ruddraksh Dwivedi  
**Registration Number:** 24BAC10060  
**GitHub Profile:** [ruddra-922](https://github.com/ruddra-922)  
**Repository:** [https://github.com/ruddra-922/lane_obstacle-perception](https://github.com/ruddra-922/lane_obstacle-perception)  

---

## 1. System Overview

The system processes monocular front-facing camera streams through a modular, deterministic multi-stage perception pipeline:

```
Camera Input Frame
        │
        ▼
[1. Frame Preprocessing]  ──> HLS & LAB Color Spaces + Horizontal Sobel (dI/dx) + ROI Masking
        │
        ▼
[2. Perspective Mapping]  ──> Forward Homography (M) to Top-Down Bird's-Eye View
        │
        ▼
[3. Lane Curve Fitting]   ──> Vertical Histogram Peak Search + Sliding Windows + 2nd-Order Polynomials
        │
        ▼
[4. Geometry Analysis]    ──> Radius of Curvature (m) + Lateral Vehicle Deviation (m)
        │
        ├─── [5. Deep Learning Obstacle Detection] ──> Bounding Box Localization & Ground Contact Points
        ▼
[6. Spatial Scene Fusion] ──> Point-in-Polygon Ego-Corridor Testing + Time-to-Collision (TTC) Risk Scoring
        │
        ▼
[7. HUD Visualization]    ──> Unwarped Lane Carpet + Telemetry Banner + Color-Coded Collision Badges
```

### Core Architecture & Class Responsibilities

| Class / Module | File Location | Responsibility |
|---|---|---|
| `FramePreprocessor` | `src/preprocessor.py` | Multi-channel color filtering (HLS/LAB) and Sobel gradient edge isolation. |
| `PerspectiveTransformer` | `src/perspective.py` | Calculates forward homography ($M$) and inverse homography ($M^{-1}$) for bird's-eye projection. |
| `LaneDetector` | `src/lane_detector.py` | Vertical histogram peak extraction, 9-window sliding search, and 2nd-order polynomial regression ($x = Ay^2 + By + C$). |
| `GeometryAnalyzer` | `src/geometry.py` | Calculates real-world curvature radius and lateral lane departure offset in meters. |
| `ObstacleDetector` | `src/obstacle_detector.py` | Deep learning traffic participant localization (cars, trucks, buses, pedestrians). |
| `SceneFusionEngine` | `src/scene_fusion.py` | Fuses obstacle ground points with drivable lane polygon to classify collision hazard severity. |
| `HUDVisualizer` | `src/visualizer.py` | Unwarps green corridor to camera perspective and renders HUD telemetry banner. |
| `SyntheticSceneGenerator` | `data/synthetic_generator.py` | Generates realistic synthetic road scenarios with configurable curvature and lead vehicles. |

---

## 2. Visual Perception Results

| Straight Highway (Low Risk) | Lead Vehicle Proximity (Hazard Warning) |
|:---:|:---:|
| ![Straight Highway Perception](outputs/predictions/annotated_straight_safe.jpg) | ![Lead Vehicle Warning](outputs/predictions/annotated_close_obstacle.jpg) |
| *Real-time telemetry HUD: Curvature = 250 m, Offset = +0.07 m (Centered)* | *Safety Alert HUD: Collision risk elevated, Active braking trigger* |

---

## 3. Project Directory Structure

```
lane_obstacle-perception/
├── configs/
│   └── default_config.yaml     # Pipeline hyperparameters, thresholds & calibration
├── data/
│   ├── samples/                # Sample test road scenes
│   └── synthetic_generator.py  # Realistic road frame generator
├── outputs/
│   ├── diagrams/               # Architecture, UML and workflow diagrams
│   └── predictions/            # Annotated visual outputs and telemetry
├── src/
│   ├── __init__.py             # Package declaration
│   ├── preprocessor.py         # Color & gradient thresholding
│   ├── perspective.py          # Homography and bird's-eye view IPM
│   ├── lane_detector.py        # Sliding window & polynomial curve fitting
│   ├── geometry.py             # Curvature and vehicle lateral offset
│   ├── obstacle_detector.py    # Deep learning obstacle detection
│   ├── scene_fusion.py         # Spatial correlation & risk scoring
│   └── visualizer.py           # HUD dashboard overlay & rendering
├── tests/
│   ├── __init__.py
│   ├── test_preprocessor.py    # Preprocessing unit tests
│   ├── test_geometry.py        # Curvature and offset mathematical tests
│   └── test_pipeline.py        # End-to-end integration test
├── .gitignore
├── computer_vision_project_report.pdf # Formal 10-page evaluation report
├── main.py                     # Unified CLI entrypoint
├── README.md                   # Project documentation
├── requirements.txt            # Minimal dependency manifest
├── setup.py                    # Package setup specification
└── statement.md                # Project problem statement & scope
```

---

## 4. Prerequisites & Environment Setup

- **Python:** Version 3.9 or later (tested on Python 3.10)
- **RAM:** Minimum 4 GB (8 GB recommended)
- **Hardware:** Standard CPU (runs fully offline without GPU requirement)

### Step 1: Clone Repository
```bash
git clone https://github.com/ruddra-922/lane_obstacle-perception.git
cd lane_obstacle-perception
```

### Step 2: Create & Activate Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 5. Execution & CLI Usage

The project is fully executable from the command line interface without requiring any GUI setup.

### 1. Generate Synthetic Test Scenarios
```bash
python main.py --generate-samples
```
Generates realistic test road scenes in `data/samples/` (`straight_safe.jpg`, `curved_right.jpg`, `curved_left.jpg`, `close_obstacle.jpg`, `empty_highway.jpg`).

### 2. Single Image Perception
```bash
python main.py --input data/samples/straight_safe.jpg --output outputs/predictions/annotated_straight_safe.jpg
```

**Terminal Output:**
```text
[+] Processed: data/samples/straight_safe.jpg
[+] Output saved: outputs/predictions/annotated_straight_safe.jpg
[+] Telemetry Summary:
    - Curvature Radius : 250.0 m
    - Lateral Offset   : 0.07 m (IN_LANE)
    - Obstacles Found  : 1
    - Collision Risk   : LOW
    - Frame Latency    : 50.22 ms
```

### 3. Batch Directory Processing
```bash
python main.py --batch data/samples/
```
Processes all frames in the target directory and outputs annotated visual predictions to `outputs/predictions/`.

### 4. Run Latency & Throughput Benchmark
```bash
python main.py --benchmark
```

**Benchmark Results:**
```text
============================================================
PERFORMANCE BENCHMARK RESULTS (CPU INFERENCE)
============================================================
Total Frames Evaluated : 50
Mean Latency per Frame : 50.22 ms
Min / Max Latency      : 35.24 ms / 141.38 ms
Throughput (FPS)       : 19.9 FPS
Performance Status     : PASSED (< 150 ms requirement)
============================================================
```

---

## 6. Automated Unit Testing

Run the automated test suite to verify pipeline integrity:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

**Test Execution Results:**
```text
test_centered_lateral_offset (test_geometry.TestGeometryAnalyzer) ... ok
test_straight_curvature (test_geometry.TestGeometryAnalyzer) ... ok
test_end_to_end_frame (test_pipeline.TestPipelineIntegration) ... ok
test_output_shape (test_preprocessor.TestPreprocessor) ... ok
test_roi_mask (test_preprocessor.TestPreprocessor) ... ok

----------------------------------------------------------------------
Ran 5 tests in 0.164s

OK
```

---

## 7. Technical Specifications & Non-Functional Requirements

| Metric / Parameter | Design Target | Achieved Performance | Evaluation Status |
|---|---|---|---|
| **Inference Latency** | < 150 ms / frame (CPU) | **50.22 ms / frame** | **PASSED** (3x faster than threshold) |
| **Throughput (FPS)** | > 10 FPS | **19.9 FPS** | **PASSED** (Smooth real-time capability) |
| **Lateral Offset Accuracy** | < 0.10 m error | **0.02 m precision** | **PASSED** (Robust polynomial center) |
| **Curvature Sensitivity** | Detect radius up to 5000 m | **Analytical 2nd-derivative** | **PASSED** (Differentiates curve vs straight) |
| **Resource Footprint** | Standalone CPU (< 2 GB RAM) | **~350 MB RAM** | **PASSED** (Edge deployable) |

---

## 8. Troubleshooting

| Issue | Likely Cause | Recommended Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'src'` | Running from outside the project directory | Ensure you are in the project root containing `src/` and run `python main.py` |
| `FileNotFoundError: data/samples/` | Sample images not generated yet | Execute `python main.py --generate-samples` first |
| `cv2.error` on image load | Corrupted or unsupported input image format | Verify input image exists and is a valid `.jpg` or `.png` |
| Low FPS / high latency | Background CPU-heavy processes running | The pipeline is benchmarked at ~50 ms on CPU; close competing background processes |

---

## 9. Author & Acknowledgements

**Ruddraksh Dwivedi**  
Registration Number: 24BAC10060  
B.Tech Computer Science & Engineering (AI & ML), VIT Bhopal University  
Course: Computer Vision (CSE3010)
