"""
Unified CLI Entry Point & Autonomous Perception Pipeline
Vision-Based Autonomous Lane & Obstacle Perception Pipeline
Course: Computer Vision (CSE3010) - VIT Bhopal University
Author: Ruddraksh Dwivedi (Reg No: 24BAC10060)
"""

import argparse
import glob
import os
import sys
import time
import yaml
import cv2
import numpy as np

# Ensure src modules are resolvable if present
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Attempt modular import from src/ package; fallback to embedded implementations for standalone execution
try:
    from src.preprocessor import FramePreprocessor
    from src.perspective import PerspectiveTransformer
    from src.lane_detector import LaneDetector
    from src.geometry import GeometryAnalyzer
    from src.obstacle_detector import ObstacleDetector
    from src.scene_fusion import SceneFusionEngine
    from src.visualizer import HUDVisualizer
    from data.synthetic_generator import SyntheticSceneGenerator
except ImportError:
    # Embedded fallback implementations for standalone execution
    class FramePreprocessor:
        def __init__(self, config=None):
            cfg = config or {}
            self.ksize = cfg.get("sobel_ksize", 3)
            self.sobel_thresh = tuple(cfg.get("sobel_threshold", [25, 255]))
            self.s_thresh = tuple(cfg.get("s_channel_threshold", [120, 255]))
            self.b_thresh = tuple(cfg.get("b_channel_threshold", [140, 255]))

        def process(self, frame: np.ndarray) -> np.ndarray:
            h, w = frame.shape[:2]
            blurred = cv2.bilateralFilter(frame, d=9, sigmaColor=75, sigmaSpace=75)
            hls = cv2.cvtColor(blurred, cv2.COLOR_BGR2HLS)
            s_channel = hls[:, :, 2]
            lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
            b_channel = lab[:, :, 2]
            s_binary = np.zeros_like(s_channel, dtype=np.uint8)
            s_binary[(s_channel >= self.s_thresh[0]) & (s_channel <= self.s_thresh[1])] = 255
            b_binary = np.zeros_like(b_channel, dtype=np.uint8)
            b_binary[(b_channel >= self.b_thresh[0]) & (b_channel <= self.b_thresh[1])] = 255
            gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=self.ksize)
            abs_sobel = np.absolute(sobelx)
            max_val = np.max(abs_sobel)
            scaled_sobel = np.uint8(255 * abs_sobel / (max_val if max_val > 0 else 1))
            sobel_binary = np.zeros_like(scaled_sobel, dtype=np.uint8)
            sobel_binary[(scaled_sobel >= self.sobel_thresh[0]) & (scaled_sobel <= self.sobel_thresh[1])] = 255
            combined = np.zeros_like(s_binary, dtype=np.uint8)
            combined[(s_binary == 255) | (b_binary == 255) | (sobel_binary == 255)] = 255
            mask = np.zeros_like(combined)
            roi_vertices = np.array([[
                (int(w * 0.1), h),
                (int(w * 0.42), int(h * 0.60)),
                (int(w * 0.58), int(h * 0.60)),
                (int(w * 0.95), h)
            ]], dtype=np.int32)
            cv2.fillPoly(mask, roi_vertices, 255)
            return cv2.bitwise_and(combined, mask)

    class PerspectiveTransformer:
        def __init__(self, config=None):
            cfg = config or {}
            self.img_w = cfg.get("image_width", 1280)
            self.img_h = cfg.get("image_height", 720)
            self.src_pts = np.float32([[200, 720], [550, 470], [730, 470], [1100, 720]])
            self.dst_pts = np.float32([[300, 720], [300, 0], [980, 0], [980, 720]])
            self.M = cv2.getPerspectiveTransform(self.src_pts, self.dst_pts)
            self.Minv = cv2.getPerspectiveTransform(self.dst_pts, self.src_pts)

        def warp(self, image: np.ndarray) -> np.ndarray:
            return cv2.warpPerspective(image, self.M, (self.img_w, self.img_h), flags=cv2.INTER_LINEAR)

        def unwarp(self, image: np.ndarray) -> np.ndarray:
            return cv2.warpPerspective(image, self.Minv, (self.img_w, self.img_h), flags=cv2.INTER_LINEAR)

    class LaneDetector:
        def __init__(self, config=None):
            cfg = config or {}
            self.nwindows = cfg.get("nwindows", 9)
            self.margin = cfg.get("margin", 80)
            self.minpix = cfg.get("minpix", 40)

        def fit_polynomial(self, binary_warped: np.ndarray):
            h, w = binary_warped.shape[:2]
            histogram = np.sum(binary_warped[h // 2:, :], axis=0)
            midpoint = w // 2
            leftx_base = np.argmax(histogram[:midpoint])
            rightx_base = np.argmax(histogram[midpoint:]) + midpoint
            if leftx_base == 0: leftx_base = int(w * 0.25)
            if rightx_base == midpoint: rightx_base = int(w * 0.75)
            window_height = h // self.nwindows
            nonzero = binary_warped.nonzero()
            nonzeroy, nonzerox = np.array(nonzero[0]), np.array(nonzero[1])
            leftx_current, rightx_current = leftx_base, rightx_base
            left_lane_inds, right_lane_inds = [], []
            for window in range(self.nwindows):
                win_y_low = h - (window + 1) * window_height
                win_y_high = h - window * window_height
                win_xleft_low = leftx_current - self.margin
                win_xleft_high = leftx_current + self.margin
                win_xright_low = rightx_current - self.margin
                win_xright_high = rightx_current + self.margin
                good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                                  (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
                good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                                   (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
                left_lane_inds.append(good_left_inds)
                right_lane_inds.append(good_right_inds)
                if len(good_left_inds) > self.minpix:
                    leftx_current = int(np.mean(nonzerox[good_left_inds]))
                if len(good_right_inds) > self.minpix:
                    rightx_current = int(np.mean(nonzerox[good_right_inds]))
            left_lane_inds = np.concatenate(left_lane_inds) if len(left_lane_inds) > 0 else np.array([], dtype=np.int64)
            right_lane_inds = np.concatenate(right_lane_inds) if len(right_lane_inds) > 0 else np.array([], dtype=np.int64)
            ploty = np.linspace(0, h - 1, h)
            if len(left_lane_inds) >= 50:
                left_fit = np.polyfit(nonzeroy[left_lane_inds], nonzerox[left_lane_inds], 2)
            else:
                left_fit = np.array([0.0, 0.0, float(leftx_base)])
            if len(right_lane_inds) >= 50:
                right_fit = np.polyfit(nonzeroy[right_lane_inds], nonzerox[right_lane_inds], 2)
            else:
                right_fit = np.array([0.0, 0.0, float(rightx_base)])
            left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
            right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
            return left_fit, right_fit, left_fitx, right_fitx, ploty

    class GeometryAnalyzer:
        def __init__(self, config=None):
            cfg = config or {}
            self.ym_per_pix = cfg.get("ym_per_pix", 30.0 / 720.0)
            self.xm_per_pix = cfg.get("xm_per_pix", 3.7 / 700.0)

        def compute_curvature(self, left_fit, right_fit, h=720):
            y_eval = h - 1
            curvatures = []
            for fit in [left_fit, right_fit]:
                A = fit[0] * (self.xm_per_pix / (self.ym_per_pix ** 2))
                B = fit[1] * (self.xm_per_pix / self.ym_per_pix)
                if abs(A) < 1e-7:
                    curvatures.append(5000.0)
                else:
                    denom = (1 + (2 * A * y_eval * self.ym_per_pix + B) ** 2) ** 1.5
                    curvatures.append(round(denom / (2 * abs(A)), 1))
            return float(np.mean(curvatures))

        def compute_lateral_offset(self, left_fitx, right_fitx, image_width=1280):
            lane_center_px = (left_fitx[-1] + right_fitx[-1]) / 2.0
            vehicle_center_px = image_width / 2.0
            offset_m = round((vehicle_center_px - lane_center_px) * self.xm_per_pix, 2)
            if abs(offset_m) < 0.15:
                status = "IN_LANE"
            elif offset_m > 0:
                status = "DEVIATING_RIGHT"
            else:
                status = "DEVIATING_LEFT"
            return offset_m, status

    class ObstacleDetector:
        def __init__(self, config=None):
            cfg = config or {}
            self.conf_thresh = cfg.get("conf_threshold", 0.35)
            self.model = None
            try:
                from ultralytics import YOLO
                self.model = YOLO("yolov8n.pt")
            except Exception:
                self.model = None

        def detect(self, frame: np.ndarray):
            h, w = frame.shape[:2]
            detections = []
            if self.model is not None:
                try:
                    results = self.model(frame, verbose=False, conf=self.conf_thresh)
                    for r in results:
                        for box in r.boxes:
                            cls_id = int(box.cls[0].item())
                            conf = float(box.conf[0].item())
                            if cls_id in [2, 3, 5, 7, 0]:
                                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                                detections.append({
                                    "bbox": xyxy.tolist(),
                                    "class": self.model.names[cls_id],
                                    "confidence": round(conf, 2),
                                    "bottom_center": [int((xyxy[0] + xyxy[2]) / 2), int(xyxy[3])]
                                })
                    if detections:
                        return detections
                except Exception:
                    pass
            # Classical heuristic fallback detector
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            roi_y_start = int(h * 0.50)
            roi_y_end = int(h * 0.90)
            roi = gray[roi_y_start:roi_y_end, int(w * 0.2):int(w * 0.8)]
            blurred = cv2.GaussianBlur(roi, (7, 7), 0)
            edges = cv2.Canny(blurred, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if 800 < area < 40000:
                    bx, by, bw, bh = cv2.boundingRect(cnt)
                    aspect = bw / float(bh)
                    if 0.5 < aspect < 3.5:
                        abs_x1 = bx + int(w * 0.2)
                        abs_y1 = by + roi_y_start
                        abs_x2 = abs_x1 + bw
                        abs_y2 = abs_y1 + bh
                        detections.append({
                            "bbox": [abs_x1, abs_y1, abs_x2, abs_y2],
                            "class": "vehicle",
                            "confidence": 0.85,
                            "bottom_center": [int((abs_x1 + abs_x2) / 2), abs_y2]
                        })
            return detections

    class SceneFusionEngine:
        def __init__(self, config=None):
            pass

        def evaluate_risk(self, detections, left_fitx, right_fitx, ploty):
            h = len(ploty)
            fused_dets = []
            max_risk_level = "LOW"
            risk_priority = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
            for det in detections:
                bc_x, bc_y = det["bottom_center"]
                y_idx = int(np.clip(bc_y, 0, h - 1))
                left_edge = left_fitx[y_idx]
                right_edge = right_fitx[y_idx]
                in_lane = (left_edge - 30 <= bc_x <= right_edge + 30)
                det["in_lane"] = in_lane
                if in_lane:
                    if bc_y > int(h * 0.82):
                        risk = "CRITICAL"
                    elif bc_y > int(h * 0.70):
                        risk = "HIGH"
                    else:
                        risk = "MEDIUM"
                else:
                    risk = "LOW"
                det["risk"] = risk
                fused_dets.append(det)
                if risk_priority[risk] > risk_priority[max_risk_level]:
                    max_risk_level = risk
            return fused_dets, max_risk_level

    class HUDVisualizer:
        def __init__(self, config=None):
            pass

        def render(self, frame, left_fitx, right_fitx, ploty, transformer,
                   curvature, offset, lane_status, detections, overall_risk):
            h, w = frame.shape[:2]
            warp_zero = np.zeros((h, w), dtype=np.uint8)
            color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
            pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
            pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
            pts = np.hstack((pts_left, pts_right))
            lane_color = (0, 220, 100)
            if overall_risk in ["HIGH", "CRITICAL"]:
                lane_color = (0, 140, 255)
            cv2.fillPoly(color_warp, np.int_([pts]), lane_color)
            cv2.polylines(color_warp, np.int_([pts_left]), False, (255, 255, 0), 20)
            cv2.polylines(color_warp, np.int_([pts_right]), False, (0, 255, 255), 20)
            newwarp = transformer.unwarp(color_warp)
            result = cv2.addWeighted(frame, 1.0, newwarp, 0.35, 0)
            # Render obstacle bounding boxes
            for det in detections:
                bx1, by1, bx2, by2 = det["bbox"]
                box_color = (0, 255, 0) if det.get("risk") == "LOW" else (0, 69, 255)
                cv2.rectangle(result, (bx1, by1), (bx2, by2), box_color, 3)
                label = f"{det['class']} [{det.get('risk', 'LOW')}]"
                cv2.putText(result, label, (bx1, max(20, by1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            # HUD Telemetry Overlay
            overlay = result.copy()
            cv2.rectangle(overlay, (20, 20), (520, 160), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.75, result, 0.25, 0, result)
            cv2.rectangle(result, (20, 20), (520, 160), (0, 200, 255), 2)
            cv2.putText(result, "AUTONOMOUS PERCEPTION PIPELINE", (35, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2)
            cv2.putText(result, f"Curvature Radius : {curvature:.1f} m", (35, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.putText(result, f"Lateral Offset   : {offset:+.2f} m ({lane_status})", (35, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.putText(result, f"Obstacles Detected: {len(detections)}", (35, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            risk_color = (0, 255, 0)
            if overall_risk == "MEDIUM": risk_color = (0, 255, 255)
            elif overall_risk in ["HIGH", "CRITICAL"]: risk_color = (0, 0, 255)
            cv2.putText(result, f"HAZARD: {overall_risk}", (350, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, risk_color, 2)
            return result

    class SyntheticSceneGenerator:
        def __init__(self, width=1280, height=720):
            self.w = width
            self.h = height

        def generate(self, curve_direction="straight", has_lead_vehicle=False, vehicle_distance="far"):
            frame = np.zeros((self.h, self.w, 3), dtype=np.uint8)
            frame[:int(self.h * 0.45), :] = (135, 90, 60)
            frame[int(self.h * 0.45):, :] = (45, 45, 45)
            # Center road perspective
            pts_road = np.array([
                [int(self.w * 0.45), int(self.h * 0.45)],
                [int(self.w * 0.55), int(self.h * 0.45)],
                [self.w, self.h],
                [0, self.h]
            ], dtype=np.int32)
            cv2.fillPoly(frame, [pts_road], (60, 60, 60))
            # Lane markings
            cv2.line(frame, (int(self.w * 0.45), int(self.h * 0.45)), (int(self.w * 0.15), self.h), (255, 255, 255), 12)
            cv2.line(frame, (int(self.w * 0.55), int(self.h * 0.45)), (int(self.w * 0.85), self.h), (0, 220, 255), 12)
            # Dashed center line
            for y in range(int(self.h * 0.48), self.h, 45):
                interp = (y - self.h * 0.45) / (self.h * 0.55)
                cx = int(self.w * 0.50 + (interp * 15))
                cv2.line(frame, (cx, y), (cx, min(y + 25, self.h)), (255, 255, 255), 6)
            if has_lead_vehicle:
                vy = int(self.h * 0.72) if vehicle_distance == "close" else int(self.h * 0.55)
                scale = 1.6 if vehicle_distance == "close" else 0.8
                vx = int(self.w * 0.50)
                bw, bh = int(140 * scale), int(90 * scale)
                cv2.rectangle(frame, (vx - bw//2, vy - bh), (vx + bw//2, vy), (20, 20, 200), -1)
                cv2.rectangle(frame, (vx - bw//2, vy - bh), (vx + bw//2, vy), (255, 255, 255), 2)
            return frame

        def generate_test_suite(self, output_dir="data/samples"):
            os.makedirs(output_dir, exist_ok=True)
            samples = {
                "straight_safe.jpg": self.generate(curve_direction="straight", has_lead_vehicle=True, vehicle_distance="far"),
                "curved_right.jpg": self.generate(curve_direction="right", has_lead_vehicle=False),
                "curved_left.jpg": self.generate(curve_direction="left", has_lead_vehicle=False),
                "close_obstacle.jpg": self.generate(curve_direction="straight", has_lead_vehicle=True, vehicle_distance="close"),
                "empty_highway.jpg": self.generate(curve_direction="straight", has_lead_vehicle=False)
            }
            paths = {}
            for name, img in samples.items():
                p = os.path.join(output_dir, name)
                cv2.imwrite(p, img)
                paths[name] = p
            return paths


class AutonomousPerceptionPipeline:
    def __init__(self, config_path=None):
        if config_path and os.path.exists(config_path):
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {}

        self.preprocessor = FramePreprocessor(self.config)
        self.transformer = PerspectiveTransformer(self.config)
        self.lane_detector = LaneDetector(self.config)
        self.geometry = GeometryAnalyzer(self.config)
        self.obstacle_detector = ObstacleDetector(self.config)
        self.fusion_engine = SceneFusionEngine(self.config)
        self.visualizer = HUDVisualizer(self.config)

    def process_frame(self, frame: np.ndarray):
        """Execute single frame perception and telemetry inference."""
        t0 = time.perf_counter()

        # 1. Preprocessing & Gradient Filtering
        binary_lane = self.preprocessor.process(frame)

        # 2. Perspective Homography (IPM)
        binary_warped = self.transformer.warp(binary_lane)

        # 3. Sliding Window Polynomial Fitting
        left_fit, right_fit, left_fitx, right_fitx, ploty = self.lane_detector.fit_polynomial(binary_warped)

        # 4. Metric Geometry Calculations
        h, w = frame.shape[:2]
        curvature = self.geometry.compute_curvature(left_fit, right_fit, h=h)
        offset, lane_status = self.geometry.compute_lateral_offset(left_fitx, right_fitx, image_width=w)

        # 5. Deep Learning Obstacle Detection
        detections = self.obstacle_detector.detect(frame)

        # 6. Spatial Scene Fusion & Collision Risk Assessment
        fused_dets, overall_risk = self.fusion_engine.evaluate_risk(detections, left_fitx, right_fitx, ploty)

        # 7. Render Visualization & HUD Dashboard
        annotated_frame = self.visualizer.render(
            frame, left_fitx, right_fitx, ploty, self.transformer,
            curvature, offset, lane_status, fused_dets, overall_risk
        )

        latency_ms = (time.perf_counter() - t0) * 1000.0

        telemetry = {
            "curvature_radius_m": curvature,
            "lateral_offset_m": offset,
            "lane_status": lane_status,
            "obstacle_count": len(fused_dets),
            "overall_risk": overall_risk,
            "latency_ms": latency_ms
        }

        return annotated_frame, telemetry


def main():
    parser = argparse.ArgumentParser(
        description="Autonomous Lane and Obstacle Perception Pipeline (CSE3010)"
    )
    parser.add_argument("--input", type=str, help="Path to single input image")
    parser.add_argument("--output", type=str, default="outputs/predictions/output.jpg", help="Path to save output image")
    parser.add_argument("--batch", type=str, help="Directory containing batch of images to process")
    parser.add_argument("--generate-samples", action="store_true", help="Generate synthetic test road frames")
    parser.add_argument("--benchmark", action="store_true", help="Run 50-frame latency and FPS benchmark")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="Configuration file path")

    args = parser.parse_args()

    pipeline = AutonomousPerceptionPipeline(config_path=args.config)

    # Mode 1: Generate synthetic test samples
    if args.generate_samples:
        print("[+] Generating synthetic road test samples...")
        gen = SyntheticSceneGenerator()
        paths = gen.generate_test_suite(output_dir="data/samples")
        for name, p in paths.items():
            print(f"    Saved: {os.path.abspath(p)}")
        print("[+] Sample generation complete!")
        return

    # Mode 2: Benchmark Latency & Throughput
    if args.benchmark:
        print("[*] Benchmarking pipeline performance across 50 simulated frames...")
        gen = SyntheticSceneGenerator()
        test_frame = gen.generate()
        latencies = []
        for i in range(50):
            t0 = time.perf_counter()
            _, _ = pipeline.process_frame(test_frame)
            latencies.append((time.perf_counter() - t0) * 1000.0)

        mean_lat = np.mean(latencies)
        fps = 1000.0 / mean_lat if mean_lat > 0 else 0
        print("=" * 60)
        print("PERFORMANCE BENCHMARK RESULTS (CPU INFERENCE)")
        print("=" * 60)
        print(f"Total Frames Evaluated : {len(latencies)}")
        print(f"Mean Latency per Frame : {mean_lat:.2f} ms")
        print(f"Min / Max Latency      : {np.min(latencies):.2f} ms / {np.max(latencies):.2f} ms")
        print(f"Throughput (FPS)       : {fps:.1f} FPS")
        print(f"Performance Status     : {'PASSED (< 150 ms requirement)' if mean_lat < 150 else 'EXCEEDED'}")
        print("=" * 60)
        return

    # Mode 3: Batch Directory Processing
    if args.batch:
        if not os.path.exists(args.batch):
            print(f"[-] Error: Target directory '{args.batch}' not found.")
            sys.exit(1)

        img_paths = glob.glob(os.path.join(args.batch, "*.[jJ][pP][gG]")) +                     glob.glob(os.path.join(args.batch, "*.[pP][nN][gG]"))
        print(f"[*] Processing {len(img_paths)} frames from directory: {args.batch}")
        os.makedirs("outputs/predictions", exist_ok=True)

        for p in img_paths:
            frame = cv2.imread(p)
            if frame is None:
                continue
            annotated, tel = pipeline.process_frame(frame)
            fname = os.path.basename(p)
            out_p = os.path.join("outputs/predictions", f"annotated_{fname}")
            cv2.imwrite(out_p, annotated)
            print(f"    Processed {fname:25s} -> {out_p} (Lat: {tel['latency_ms']:.1f}ms, Risk: {tel['overall_risk']})")
        print("[+] Batch processing completed successfully!")
        return

    # Mode 4: Single Image Inference
    if args.input:
        if not os.path.exists(args.input):
            print(f"[-] Error: Input file '{args.input}' does not exist.")
            sys.exit(1)

        frame = cv2.imread(args.input)
        if frame is None:
            print(f"[-] Error: Could not decode image at '{args.input}'.")
            sys.exit(1)

        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        annotated, telemetry = pipeline.process_frame(frame)
        cv2.imwrite(args.output, annotated)

        print(f"[+] Processed: {args.input}")
        print(f"[+] Output saved: {args.output}")
        print("[+] Telemetry Summary:")
        print(f"    - Curvature Radius : {telemetry['curvature_radius_m']} m")
        print(f"    - Lateral Offset   : {telemetry['lateral_offset_m']} m ({telemetry['lane_status']})")
        print(f"    - Obstacles Found  : {telemetry['obstacle_count']}")
        print(f"    - Collision Risk   : {telemetry['overall_risk']}")
        print(f"    - Frame Latency    : {telemetry['latency_ms']:.2f} ms")
        return

    # Default fallback: show help
    parser.print_help()


if __name__ == "__main__":
    main()
