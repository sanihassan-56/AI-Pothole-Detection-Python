from ultralytics import YOLO

# 1. Load your trained model weights
model = YOLO("best.pt")

# 2. Export to ONNX with INT8 quantization
# The 'data' parameter is required for INT8 quantization calibration using your dataset
model.export(
    format="onnx", 
    int8=True, 
    data="data.yaml",  # Update with your actual path to data.yaml
    imgsz=640
)

print("Model successfully exported and quantized to ONNX INT8 format!")