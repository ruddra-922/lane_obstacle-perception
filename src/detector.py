import cv2
import numpy as np

class LaneObstacleDetector:
    def __init__(self):
        try:
            from ultralytics import YOLO
            self.model = YOLO("yolov8n.pt")
        except Exception:
            self.model = None

    def detect_and_estimate(self, image_path):
        """Run perception on road scene."""
        from main import AutonomousPerceptionPipeline
        pipeline = AutonomousPerceptionPipeline()
        frame = cv2.imread(image_path)
        if frame is not None:
            annotated, telemetry = pipeline.process_frame(frame)
            return annotated, telemetry
        return None, None
