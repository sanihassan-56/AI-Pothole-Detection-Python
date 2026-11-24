
import os
import cv2
import tkinter as tk
from tkinter import filedialog
from ultralytics import YOLO

# ==========================================
# 1. INITIALIZE GUI SELECTION MENU
# ==========================================

selected_mode = None

def set_mode(mode):
    global selected_mode
    selected_mode = mode
    root.destroy()

root = tk.Tk()
root.title("AI Road Defect Detection Suite")
root.geometry("340x260")
root.resizable(False, False)

tk.Label(
    root,
    text="Select Detection Mode",
    font=("Arial", 13, "bold")
).pack(pady=15)

tk.Button(
    root,
    text="🖼️ Test Image from Disk",
    width=26,
    height=2,
    bg="#e0e0e0",
    command=lambda: set_mode("image")
).pack(pady=5)

tk.Button(
    root,
    text="🎬 Test Video File from Disk",
    width=26,
    height=2,
    bg="#e0e0e0",
    command=lambda: set_mode("video")
).pack(pady=5)

tk.Button(
    root,
    text="📹 Real-Time Webcam Live Feed",
    width=26,
    height=2,
    bg="#e0e0e0",
    command=lambda: set_mode("webcam")
).pack(pady=5)

root.mainloop()

if not selected_mode:
    print("No mode selected. Exiting script.")
    raise SystemExit

# ==========================================
# 2. LOAD YOLO MODEL
# ==========================================

model = YOLO("best.onnx")

color_map = {
    "pothole": (0, 0, 255),
    "cracks": (255, 0, 0),
    "patching": (0, 255, 255)
}

# ==========================================
# 3. ANNOTATION FUNCTION
# ==========================================

def annotate_frame(frame, results):

    h, w = frame.shape[:2]
    base_dim = min(w, h)

    dynamic_scale = max(0.4, base_dim / 900.0)
    font_scale = dynamic_scale * 0.75
    font_thickness = max(1, int(dynamic_scale * 2))
    box_thickness = max(2, int(dynamic_scale * 3))

    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            class_name = model.names[class_id]

            x1, y1, x2, y2 = map(
                int, box.xyxy[0].tolist()
            )

            box_color = color_map.get(
                class_name.lower(), (0, 255, 0)
            )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                box_color,
                box_thickness
            )

            conf_percent = int(round(confidence * 100))
            label = f"{class_name} {conf_percent}%"

            (text_width, text_height), _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                font_thickness
            )

            text_bg_y1 = max(
                y1 - text_height - int(12 * dynamic_scale), 0
            )

            text_bg_y2 = y1
            text_bg_x1 = x1
            text_bg_x2 = x1 + text_width + int(10 * dynamic_scale)

            cv2.rectangle(
                frame,
                (text_bg_x1, text_bg_y1),
                (text_bg_x2, text_bg_y2),
                (115, 115, 0),
                -1
            )

            text_org = (
                x1 + int(5 * dynamic_scale),
                y1 - int(6 * dynamic_scale)
            )

            cv2.putText(
                frame,
                label,
                text_org,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                font_thickness
            )

    # Main heading banner

    heading_text = "AI POTHOLE DETECTION MODEL"
    heading_scale = dynamic_scale * 0.85
    heading_thickness = max(1, int(dynamic_scale * 2))

    (h_width, h_height), _ = cv2.getTextSize(
        heading_text,
        cv2.FONT_HERSHEY_SIMPLEX,
        heading_scale,
        heading_thickness
    )

    pad = int(12 * dynamic_scale)

    hx1 = int(15 * dynamic_scale)
    hy1 = int(15 * dynamic_scale)
    hx2 = hx1 + h_width + (pad * 2)
    hy2 = hy1 + h_height + (pad * 2)

    cv2.rectangle(
        frame, (hx1, hy1), (hx2, hy2), (50, 50, 50), -1
    )

    cv2.rectangle(
        frame,
        (hx1, hy1),
        (hx2, hy2),
        (200, 200, 200),
        max(1, int(dynamic_scale))
    )

    cv2.putText(
        frame,
        heading_text,
        (
            hx1 + pad,
            hy1 + h_height + int(pad * 0.6)
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        heading_scale,
        (255, 255, 255),
        heading_thickness
    )

    return frame


# ==========================================
# 4. MODE 1: IMAGE DETECTION
# ==========================================

if selected_mode == "image":

    root = tk.Tk()
    root.withdraw()

    image_path = filedialog.askopenfilename(
        title="Select Test Image",
        filetypes=[
            ("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp")
        ]
    )

    root.destroy()

    if not image_path:
        print("No image selected.")
        raise SystemExit

    print(f"Processing image: {image_path}")

    results = model.predict(
        source=image_path,
        conf=0.25,
        imgsz=640,
        save=False
    )

    # Model information

    print("\nMODEL INFORMATION")
    print("Task:", model.task)
    print("Classes:", model.names)

    print("\nDETECTION RESULTS")

    for result in results:

        if result.boxes is None or len(result.boxes) == 0:
            print("No defects detected.")
            continue

        for box in result.boxes:

            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            coordinates = box.xyxy[0].tolist()

            print("Class ID:", class_id)
            print("Class name:", model.names[class_id])
            print("Confidence:", round(confidence, 4))
            print("Bounding box:", coordinates)
            print("--------------------")

    img = cv2.imread(image_path)

    if img is None:
        print("Error: Could not read image.")
        raise SystemExit

    annotated_img = annotate_frame(img, results)

    output_dir = "runs/detect/outputs/predict"
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(
        output_dir, "latest_result.jpg"
    )

    cv2.imwrite(output_path, annotated_img)

    print(f"Saved result to: {output_path}")

    cv2.imshow(
        "Road Defect Detection - Image Result",
        annotated_img
    )

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ==========================================
# 5. MODE 2: VIDEO DETECTION
# ==========================================

elif selected_mode == "video":

    root = tk.Tk()
    root.withdraw()

    video_path = filedialog.askopenfilename(
        title="Select Test Video",
        filetypes=[
            ("Video Files", "*.mp4 *.avi *.mov *.mkv")
        ]
    )

    root.destroy()

    if not video_path:
        print("No video selected.")
        raise SystemExit

    print(f"Processing video: {video_path}")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video file.")
        raise SystemExit

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_dir = "runs/detect/outputs/predict"
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(
        output_dir, "output_video.mp4"
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height)
    )

    if not out.isOpened():
        print("Error: Could not create output video.")
        cap.release()
        raise SystemExit

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        results = model.predict(
            source=frame,
            conf=0.25,
            imgsz=640,
            verbose=False
        )

        annotated_frame = annotate_frame(
            frame, results
        )

        out.write(annotated_frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("Video processing complete!")
    print(f"Saved result to: {output_path}")


# ==========================================
# 6. MODE 3: REAL-TIME WEBCAM
# ==========================================

elif selected_mode == "webcam":

    print(
        "Starting real-time webcam detection. "
        "Click EXIT or press 'q' to quit."
    )

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        raise SystemExit

    exit_btn_coords = {
        "x1": 0, "y1": 0, "x2": 0, "y2": 0
    }

    stop_stream = False

    def mouse_click(event, x, y, flags, param):

        global stop_stream

        if event == cv2.EVENT_LBUTTONDOWN:

            if (
                exit_btn_coords["x1"] <= x <= exit_btn_coords["x2"]
                and
                exit_btn_coords["y1"] <= y <= exit_btn_coords["y2"]
            ):

                print("Exit button clicked.")
                stop_stream = True

    window_name = "Real-Time Road Defect Detection"

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_click)

    while cap.isOpened() and not stop_stream:

        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to grab webcam frame.")
            break

        results = model.predict(
            source=frame,
            conf=0.25,
            imgsz=640,
            verbose=False
        )

        annotated_frame = annotate_frame(
            frame, results
        )

        h, w = annotated_frame.shape[:2]

        btn_w, btn_h = 90, 35

        bx2 = w - 15
        by1 = 15
        bx1 = bx2 - btn_w
        by2 = by1 + btn_h

        exit_btn_coords = {
            "x1": bx1,
            "y1": by1,
            "x2": bx2,
            "y2": by2
        }

        cv2.rectangle(
            annotated_frame,
            (bx1, by1),
            (bx2, by2),
            (0, 0, 180),
            -1
        )

        cv2.rectangle(
            annotated_frame,
            (bx1, by1),
            (bx2, by2),
            (255, 255, 255),
            1
        )

        cv2.putText(
            annotated_frame,
            "EXIT",
            (bx1 + 22, by1 + 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.imshow(window_name, annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("Webcam stream closed.")

else:
    print("Invalid detection mode selected.")
