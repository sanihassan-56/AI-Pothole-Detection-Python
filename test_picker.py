import os
import cv2
import tkinter as tk
from tkinter import filedialog
from ultralytics import YOLO

# 1. Hide the main tkinter root window so only the file dialog pops up
root = tk.Tk()
root.withdraw()

print("Opening file explorer to select a test image...")

# Open file dialog to choose an image from local disk
image_path = filedialog.askopenfilename(
    title="Select Road Defect Test Image",
    filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp")]
)


# Check if user cancelled the selection
if not image_path:
    print("No image selected. Exiting script.")
    exit()

print(f"Selected image: {image_path}")

# 2. Load your model (ONNX or PT)
model = YOLO("best.onnx")

# 3. Run prediction (save=False because we are drawing custom colors using OpenCV)
results = model.predict(
    source=image_path,
    conf=0.25,
    imgsz=640,
    save=False,
)

# 4. Load the image with OpenCV
img = cv2.imread(image_path)

if img is None:
    print(f"Error: Could not load image from {image_path}")
    exit()

# Get image dimensions to compute dynamic scaling
img_height, img_width = img.shape[:2]
base_dim = min(img_width, img_height)

# Calculate dynamic scaling factor based on image resolution
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

for result in results:
    print("\nDefect predictions:")
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

        # Draw the dynamically scaled bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), box_color, box_thickness)
        
        # Prepare label text with confidence in percentage format (e.g., "pothole 85%")
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
        
        # Draw background rectangle (BGR color)
        bg_color = (115, 115, 0)
        cv2.rectangle(img, (text_bg_x1, text_bg_y1), (text_bg_x2, text_bg_y2), bg_color, -1)
        
        # Put the label text on top of the background
        text_org = (x1 + int(5 * dynamic_scale), y1 - int(6 * dynamic_scale))
        cv2.putText(img, label, text_org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)

    if not found:
        print("No pothole prediction returned.")

# 5. Draw main heading banner at the top left of the image (dynamically scaled)
heading_text = "AI POTHOLE DETECTION MODEL"
heading_scale = dynamic_scale * 0.85
heading_thickness = max(1, int(dynamic_scale * 2))

(h_width, h_height), h_baseline = cv2.getTextSize(
    heading_text, cv2.FONT_HERSHEY_SIMPLEX, heading_scale, heading_thickness
)

# Padding for the heading background
pad = int(12 * dynamic_scale)
hx1, hy1 = int(15 * dynamic_scale), int(15 * dynamic_scale)
hx2 = hx1 + h_width + (pad * 2)
hy2 = hy1 + h_height + (pad * 2)

# Draw dark banner background and border
cv2.rectangle(img, (hx1, hy1), (hx2, hy2), (50, 50, 50), -1)
cv2.rectangle(img, (hx1, hy1), (hx2, hy2), (200, 200, 200), max(1, int(dynamic_scale)))

# Put heading text inside the banner
cv2.putText(img, heading_text, (hx1 + pad, hy1 + h_height + int(pad * 0.6)), 
            cv2.FONT_HERSHEY_SIMPLEX, heading_scale, (255, 255, 255), heading_thickness)

# 6. Always save and overwrite in the single predict folder using a standardized name
output_dir = "runs/detect/outputs/predict"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "latest_result.jpg")

cv2.imwrite(output_path, img)
print(f"\nSuccessfully saved/overwritten custom-colored image to: {output_path}")

# Optionally display the resulting image in a window
cv2.imshow("Road Defect Detection Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()