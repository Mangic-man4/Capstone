import cv2
import numpy as np
import tkinter as tk
from threading import Thread
from PIL import Image, ImageTk
from ultralytics import YOLO
import torch
import time
from math import dist as euclidean

# Paths
video_path = r"Burger_video\burger_video_1.mp4"
model_path = "best.pt"

# Load YOLO model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO(model_path).to(device)

# Capture video
cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1876)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1056)

# Scaled dimensions for display
scaled_width = 960
scaled_height = 640

# FPS
fps = cap.get(cv2.CAP_PROP_FPS)

# Tracking configuration
tracked_dots = []
distance_threshold = 40
max_missing_frames = 50

# Tkinter UI
root = tk.Tk()
root.title("YOLO Object Tracking")
root.geometry(f"{scaled_width}x{scaled_height}")

canvas = tk.Canvas(root, width=scaled_width, height=scaled_height)
canvas.pack()

labels = {}
labels["object"] = tk.Label(root, text=f"Objects Detected: 0", font=("Arial", 14))
labels["object"].pack()

# Draw transparent grid
def draw_grid(frame):
    grid_color = (0, 0, 0)
    grid_interval = 80
    for x in range(grid_interval, scaled_width, grid_interval):
        cv2.line(frame, (x, 0), (x, scaled_height), grid_color, 1)
    for y in range(grid_interval, scaled_height, grid_interval):
        cv2.line(frame, (0, y), (scaled_width, y), grid_color, 1)

# Update UI loop
def update_ui():
    frame = np.ones((scaled_height, scaled_width, 3), dtype=np.uint8) * 255

    labels["object"].config(text=f"Objects Detected: {len(tracked_dots)}")
    draw_grid(frame)

    now = time.time()
    flicker = int(now * 2) % 2 == 0  # Flicker every 0.5s

    for dot in tracked_dots:
        x, y = dot["pos"]
        age = now - dot["timestamp"]
        color = (0, 255, 0) if dot["missing_count"] == 0 else (0, 128, 0)

        if age >= 15:
            if flicker:
                cv2.circle(frame, (x, y), 15, (0, 0, 255), -1)  # Red flicker
        else:
            cv2.circle(frame, (x, y), 15, color, -1)

        cv2.putText(frame, f"{age:.1f}s", (x - 20, y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 2)

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img_tk = ImageTk.PhotoImage(image=img)
    canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
    canvas.img_tk = img_tk

    root.after(50, update_ui)

# Start UI loop
root.after(50, update_ui)

# Video processing thread
def process_video():
    global tracked_dots

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (scaled_width, scaled_height))
        results = model(frame, conf=0.25)[0]

        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            detections.append((cx, cy))
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 15, (0, 255, 0), -1)

        now = time.time()
        matched_dots = []

        # Match new detections to existing dots
        for det in detections:
            matched = False
            for dot in tracked_dots:
                if euclidean(det, dot["pos"]) < distance_threshold:
                    dot["pos"] = det
                    dot["last_seen"] = now
                    dot["missing_count"] = 0
                    matched_dots.append(dot)
                    matched = True
                    break
            if not matched:
                matched_dots.append({
                    "pos": det,
                    "timestamp": now,
                    "last_seen": now,
                    "missing_count": 0
                })

        # Add back older dots that are missing but still allowed
        for dot in tracked_dots:
            if dot not in matched_dots:
                dot["missing_count"] += 1
                if dot["missing_count"] <= max_missing_frames:
                    matched_dots.append(dot)

        tracked_dots = matched_dots

        cv2.imshow("Video Feed", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == 32:
            current = cap.get(cv2.CAP_PROP_POS_FRAMES)
            cap.set(cv2.CAP_PROP_POS_FRAMES, current + int(fps * 5))

    cap.release()
    cv2.destroyAllWindows()

# Start thread and Tk UI
video_thread = Thread(target=process_video)
video_thread.start()
root.mainloop()
