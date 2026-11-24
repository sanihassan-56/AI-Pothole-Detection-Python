import os
import cv2
from ultralytics import YOLO

# 1. Load your model (ONNX or PT)
model = YOLO("best.onnx")

# 2. Open the laptop webcam (0 is the default built-in camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open laptop webcam.")
    exit()

# Get webcam frame properties for dynamic scaling
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Calculate dynamic scaling factor based on webcam resolution
base_dim = min(width, height)
dynamic_scale = max(0.4, base_dim / 900.0)
font_scale = dynamic_scale * 0.75
font_thickness = max(1, int(dynamic_scale * 2))
box_thickness = max(2, int(dynamic_scale * 3))

# Define custom colors in BGR format: (Blue, Green, Red)
color_map = {
    "pothole": (0, 0, 255),     # Red
    "cracks": (255, 0, 0),      # Blue
    "patching": (0, 255, 255)   # Yellow (fallback for patching)
}

print("Starting real-time webcam detection. Press 'q' to exit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    # Run prediction on the live camera frame
    results = model.predict(source=frame, conf=0.25, imgsz=640, verbose=False)

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            class_name = model.names[class_id]
            
            # Get box coordinates (x1, y1, x2, y2)
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # Select color based on class name
            box_color = color_map.get(class_name.lower(), (0, 255, 0))

            # Draw the dynamically scaled bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, box_thickness)
            
            # Prepare label text with percentage confidence
            conf_percent = int(round(confidence * 100))
            label = f"{class_name} {conf_percent}%"
            
            # Calculate text size to fit the background box perfectly
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )
            
            # Coordinates for the background rectangle
            text_bg_y1 = max(y1 - text_height - int(12 * dynamic_scale), 0)
            text_bg_y2 = y1
            text_bg_x1 = x1
            text_bg_x2 = x1 + text_width + int(10 * dynamic_scale)
            
            # Draw background rectangle
            bg_color = (115, 115, 0)
            cv2.rectangle(frame, (text_bg_x1, text_bg_y1), (text_bg_x2, text_bg_y2), bg_color, -1)
            
            # Put the label text on top of the background
            text_org = (x1 + int(5 * dynamic_scale), y1 - int(6 * dynamic_scale))
            cv2.putText(frame, label, text_org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)

    # 3. Draw main heading banner at the top left of the frame
    heading_text = "AI POTHOLE DETECTION MODEL"
    heading_scale = dynamic_scale * 0.85
    heading_thickness = max(1, int(dynamic_scale * 2))

    (h_width, h_height), h_baseline = cv2.getTextSize(
        heading_text, cv2.FONT_HERSHEY_SIMPLEX, heading_scale, heading_thickness
    )

    pad = int(12 * dynamic_scale)
    hx1, hy1 = int(15 * dynamic_scale), int(15 * dynamic_scale)
    hx2 = hx1 + h_width + (pad * 2)
    hy2 = hy1 + h_height + (pad * 2)

    # Draw dark banner background and border
    cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (50, 50, 50), -1)
    cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (200, 200, 200), max(1, int(dynamic_scale)))

    # Put heading text inside the banner
    cv2.putText(frame, heading_text, (hx1 + pad, hy1 + h_height + int(pad * 0.6)), 
                cv2.FONT_HERSHEY_SIMPLEX, heading_scale, (255, 255, 255), heading_thickness)

    # Display the live annotated frame
    cv2.imshow("Real-Time Road Defect Detection - Webcam", frame)

    # Press 'q' key to quit the stream
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
print("Webcam stream closed.")