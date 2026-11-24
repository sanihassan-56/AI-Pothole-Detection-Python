import os
import cv2
from ultralytics import YOLO

# 1. Load your model (you can change this to "best.onnx" if testing your exported model)
model = YOLO("best.pt")

image_path = "test_image/7.jpg"

# 2. Run prediction (save=False because we are drawing custom colors using OpenCV)
results = model.predict(
    source=image_path,
    conf=0.25,
    imgsz=640,
    save=False,
)

# 3. Load the image with OpenCV
img = cv2.imread(image_path)

if img is None:
    print(f"Error: Could not load image from {image_path}")
    exit()

# Define custom colors in BGR format: (Blue, Green, Red)
color_map = {
    "pothole": (0, 0, 255),     # Red
    "cracks": (255, 0, 0),      # Blue
    "patching": (0, 255, 255)   # Yellow (fallback for patching)
}

for result in results:
    print("\nPothole predictions:")
    found = False

    for box in result.boxes:
        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())
        class_name = model.names[class_id]
        
        # Get box coordinates (x1, y1, x2, y2)
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        # Select color based on class name (defaults to green if class not found)
        box_color = color_map.get(class_name.lower(), (0, 255, 0))

        if class_name.lower() == "pothole":
            found = True
            print(f"Pothole confidence: {confidence * 100:.1f}%")

        # Draw the thicker bounding box (thickness = 3)
        cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 3)
        
        # Prepare label text with confidence in percentage format (e.g., "pothole 85%")
        conf_percent = int(round(confidence * 100))
        label = f"{class_name} {conf_percent}%"
        font_scale = 0.7
        font_thickness = 2
        
        # Calculate text size to fit the background box perfectly
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
        )
        
        # Coordinates for the background rectangle
        text_bg_y1 = max(y1 - text_height - 10, 0)
        text_bg_y2 = y1
        text_bg_x1 = x1
        text_bg_x2 = x1 + text_width + 8
        
        # Draw background rectangle (BGR color)
        bg_color = (115, 115, 0)
        cv2.rectangle(img, (text_bg_x1, text_bg_y1), (text_bg_x2, text_bg_y2), bg_color, -1)
        
        # Put the label text on top of the background
        text_org = (x1 + 4, y1 - 6)
        cv2.putText(img, label, text_org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)

    if not found:
        print("No pothole prediction returned.")

# 4. Draw main heading banner at the top left of the image
heading_text = "AI POTHOLE DETECTION MODEL"
heading_scale = 0.8
heading_thickness = 2
(h_width, h_height), h_baseline = cv2.getTextSize(
    heading_text, cv2.FONT_HERSHEY_SIMPLEX, heading_scale, heading_thickness
)

# Padding for the heading background
pad = 10
hx1, hy1 = 15, 15
hx2 = hx1 + h_width + (pad * 2)
hy2 = hy1 + h_height + (pad * 2)

# Draw dark banner background and border
cv2.rectangle(img, (hx1, hy1), (hx2, hy2), (50, 50, 50), -1)
cv2.rectangle(img, (hx1, hy1), (hx2, hy2), (200, 200, 200), 1)

# Put heading text inside the banner
cv2.putText(img, heading_text, (hx1 + pad, hy1 + h_height + pad), 
            cv2.FONT_HERSHEY_SIMPLEX, heading_scale, (255, 255, 255), heading_thickness)

# 5. Always save and overwrite in the single predict folder
output_dir = "runs/detect/outputs/predict"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, os.path.basename(image_path))

cv2.imwrite(output_path, img)
print(f"\nSuccessfully saved/overwritten custom-colored image to: {output_path}")