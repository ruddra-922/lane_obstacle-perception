# Problem Statement & Scope Specification

**Course:** Computer Vision (CSE3010)  
**Institution:** VIT Bhopal University  
**Student:** Ruddraksh Dwivedi  
**Registration Number:** 24BAC10060  
**Project Title:** Vision-Based Autonomous Lane & Obstacle Perception Pipeline  

---

## 1. Problem Statement
Autonomous ground vehicles and Advanced Driver Assistance Systems (ADAS) depend on robust, low-latency visual perception to maintain lane centering and avoid forward collisions. Real-world unconstrained road environments introduce challenging illumination variations, curved trajectories, shadow occlusions, and dynamic traffic participants. Conventional naive approaches either rely solely on computationally prohibitive 3D LiDAR sensors or lack mathematical lane boundary modeling. 

This project solves this challenge by engineering an integrated, CPU-optimized computer vision pipeline that fuses classical geometric computer vision (color spaces, directional Sobel filtering, Inverse Perspective Mapping homography, and 2nd-degree polynomial curve fitting) with deep learning object detection to concurrently compute road curvature, vehicle lateral deviation, and collision proximity risk in real-time.

---

## 2. Scope of the Project
- Real-time ingestion and preprocessing of frontal monocular camera streams (1280x720).
- Dual-space thresholding (HLS/LAB) combined with horizontal Sobel gradient filtering for white and yellow lane marker isolation.
- Homography-based Inverse Perspective Mapping (IPM) generating top-down bird's-eye views.
- Sliding-window histogram peak search with 2nd-degree polynomial regression modeling left and right lane boundaries.
- Differential geometry estimation of road radius of curvature (meters) and vehicle lateral lane departure offset (meters).
- CPU-optimized deep learning inference for dynamic obstacle detection (cars, trucks, buses, pedestrians).
- Spatial scene fusion correlating obstacle ground-contact coordinates with the ego-lane corridor to classify Time-to-Collision (TTC) hazard levels (NORMAL, ATTENTION, WARNING, CRITICAL).
- Telemetry Head-Up Display (HUD) rendering with real-time driving metrics.

---

## 3. Target Users
1. **Autonomous Vehicle / ADAS Systems**: Embedded perception stacks requiring real-time lane keeping assist (LKA) and forward collision warning (FCW).
2. **Robotics Researchers & Students**: Educational testbed for classical vs. deep learning computer vision fusion.
3. **Fleet Telematics & Safety Systems**: Post-hoc analysis of driver lane discipline and headway hazard metrics from dashcam footage.

---

## 4. High-Level Features
- **Deterministic Multi-Stage Perception**: Modular separation of pre-processing, homography, curve fitting, geometry, object detection, and HUD visualization.
- **Sub-150ms CPU Execution**: Tailored for resource-constrained edge hardware without requiring discrete GPUs.
- **Comprehensive CLI Interface**: Flexible flags for single image inference, batch directory processing, synthetic scenario generation, and latency benchmarking.
- **Robustness Against Lighting Shifts**: Combined HLS L/S channel and LAB B channel thresholding resilient to asphalt reflections and shadows.
