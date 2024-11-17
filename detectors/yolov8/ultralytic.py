"""
Ultralytics YOLOv8 model for object detection
"""
import torch
from ultralytics import YOLO

class YOLOv8():
    def __init__(self, config):
        self.model = YOLO(config)
        self.device = torch.device("cuda:0")

    def train(self, data_yaml, epochs, batch, device):
        self.model.train(data=data_yaml,
                         epochs=epochs,
                         batch=batch,
                         device=device)

    def predict(self, image_path):
        return self.model(image_path)
    
    def export(self):
        self.model.info(verbose=True)
        self.model.export(format="onnx", dynamic=False, opset=12, batch=1)
        

if __name__ == "__main__":
    yolov8 = YOLOv8("./weights/vehicle_yolov8s_640.pt")
    yolov8.export()


