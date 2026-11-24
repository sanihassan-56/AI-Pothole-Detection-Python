
import os
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from ultralytics import YOLO

# 1. Initialize GUI Selection Menu
selected_mode = None

def set_mode(mode):
    global selected_mode
    selected_mode = mode
    root.destroy()

root = tk.Tk()
root.title("AI Road Defect Detection Suite")
root.geometry("340x260")
root.resizable(False, False)

# Styling the popup menu
tk.Label(root, text="Select Detection Mode", font=("Arial", 13, "bold")).pack(pady=15)

btn_image = tk.Button(root, text="🖼️ Test Image from Disk", width=26, height=2, bg="#e0e0e0", command=lambda: set_mode("image"))
btn_image.pack(pady=5)

btn_video = tk.Button(root, text="🎬 Test Video File from Disk", width=26, height=2, bg="#e0e0e0", command=lambda: set_mode("video"))
btn_video.pack(pady=5)

btn_webcam = tk.Button(root, text="📹 Real-Time Webcam Live Feed", width=26, height=2, bg="#e0e0e0", command=lambda: set_mode("webcam"))
btn_webcam.pack(pady=5)

root.mainloop()

# If user closed the window without selecting
if not selected_mode:
    print("No mode selected. Exiting script.")
    exit()

# 2. Load your model
model = YOLO("best.onnx")

# Define custom colors in BGR format: (Blue, Green, Red)
color_map = {
    "pothole": (0, 0, 255),     # Red
    "cracks": (255, 0, 0),      # Blue
    "patching": (0, 255, 255)   # Yellow (fallback for patching)
}

# Helper function to draw all custom annotations on any frame/image
def annotate_frame(frame, results):
    h, w = frame.shape[:2]
    base_dim = min(w, h)
    
    # Calculate dynamic scaling factor based on resolution
    dynamic_scale = max(0.4, base_dim / 900.0)
    font_scale = dynamic_scale * 0.75
    font_thickness = max(1, int(dynamic_scale * 2))
    box_thickness = max(2, int(dynamic_scale * 3))

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            class_name = model.names[class_id]
            
            # Get box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # Select color based on class name
            box_color = color_map.get(class_name.lower(), (0, 255, 0))

            # Draw dynamically scaled bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, box_thickness)
            
            # Prepare label with percentage confidence
            conf_percent = int(round(confidence * 100))
            label = f"{class_name} {conf_percent}%"
            
            # Calculate text size for background box
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )
            
            text_bg_y1 = max(y1 - text_height - int(12 * dynamic_scale), 0)
            text_bg_y2 = y1
            text_bg_x1 = x1
            text_bg_x2 = x1 + text_width + int(10 * dynamic_scale)
            
            # Draw text background rectangle
            cv2.rectangle(frame, (text_bg_x1, text_bg_y1), (text_bg_x2, text_bg_y2), (115, 115, 0), -1)
            
            # Put label text
            text_org = (x1 + int(5 * dynamic_scale), y1 - int(6 * dynamic_scale))
            cv2.putText(frame, label, text_org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)

    # Draw main heading banner at the top left
    heading_text = "AI POTHOLE DETECTION MODEL"
    heading_scale = dynamic_scale * 0.85
    heading_thickness = max(1, int(dynamic_scale * 2))

    (h_width, h_height), _ = cv2.getTextSize(
        heading_text, cv2.FONT_HERSHEY_SIMPLEX, heading_scale, heading_thickness
    )

    pad = int(12 * dynamic_scale)
    hx1, hy1 = int(15 * dynamic_scale), int(15 * dynamic_scale)
    hx2 = hx1 + h_width + (pad * 2)
    hy2 = hy1 + h_height + (pad * 2)

    cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (50, 50, 50), -1)
    cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (200, 200, 200), max(1, int(dynamic_scale)))
    cv2.putText(frame, heading_text, (hx1 + pad, hy1 + h_height + int(pad * 0.6)), 
                cv2.FONT_HERSHEY_SIMPLEX, heading_scale, (255, 255, 255), heading_thickness)
    
    return frame


# ==========================================
# MODE 1: TEST IMAGE FILE
# ==========================================
if selected_mode == "image":
    root = tk.Tk()
    root.withdraw()
    image_path = filedialog.askopenfilename(
        title="Select Test Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp")]
    )
    
    if not image_path:
        print("No image selected.")
        exit()

    print(f"Processing image: {image_path}")
    results = model.predict(source=image_path, conf=0.25, imgsz=640, save=False)
    
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not read image.")
        exit()

    annotated_img = annotate_frame(img, results)

    output_dir = "runs/detect/outputs/predict"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "latest_result.jpg")

    cv2.imwrite(output_path, annotated_img)
    print(f"Saved result to: {output_path}")

    cv2.imshow("Road Defect Detection - Image Result", annotated_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ==========================================
# MODE 2: TEST VIDEO FILE
# ==========================================
elif selected_mode == "video":
    root = tk.Tk()
    root.withdraw()
    video_path = filedialog.askopenfilename(
        title="Select Test Video",
        filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")]
    )
    
    if not video_path:
        print("No video selected.")
        exit()

    print(f"Processing video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        exit()

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_dir = "runs/detect/outputs/predict"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "output_video.mp4")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, conf=0.25, imgsz=640, verbose=False)
        annotated_frame = annotate_frame(frame, results)
        out.write(annotated_frame)

    cap.release()
    out.release()
    print(f"Video processing complete! Saved to: {output_path}")


# ==========================================
# MODE 3: REAL-TIME WEBCAM
# ==========================================
elif selected_mode == "webcam":
    print("Starting real-time webcam detection. Press 'q' to exit.")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab webcam frame.")
            break

        results = model.predict(source=frame, conf=0.25, imgsz=640, verbose=False)
        annotated_frame = annotate_frame(frame, results)

        cv2.imshow("Real-Time Road Defect Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam stream closed.")