from abc import ABC, abstractmethod

import numpy as np
from ultralytics import YOLO


class Detector(ABC):
    @abstractmethod
    def predict(self, image: np.ndarray):
        pass

class YOLOv8(Detector):
    def __init__(self, detector_model_path):
        self.detector_model_path = detector_model_path
        self.model = YOLO(self.detector_model_path)
    
    def predict(self, image, project = None, name = None):
        result = self.model.predict(image, save=True, imgsz=1280, conf=0.4, project=project, name=name)
        return result