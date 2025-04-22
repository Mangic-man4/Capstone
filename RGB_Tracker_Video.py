import cv2
import numpy as np
from ultralytics import YOLO
import os
import glob
from datetime import datetime

# === Configuration ===
dot_radius = 4
dot_color_rgb = (255, 0, 0)
dot_color_bgr = (dot_color_rgb[2], dot_color_rgb[1], dot_color_rgb[0])
font_scale = 1.2
model_path = "best.pt"
video_folder = "Burger_video"
log_file = "patty_rgb_log.txt"

model = YOLO(model_path)
video_paths = sorted(glob.glob(os.path.join(video_folder, "*.mp4")))

if not video_paths:
    print(f"❌ No videos found in folder: {video_folder}")
    exit()

def average_bgr(image, box):
    x1, y1, x2, y2 = map(int, box)
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return (0, 0, 0)
    mean_bgr = np.mean(roi.astype(np.float32), axis=(0, 1))
    return tuple(map(int, mean_bgr))

# Open log file
log = open(log_file, "w")

# === Loop through videos ===
for video_path in video_paths:
    cap = cv2.VideoCapture(video_path)
    print(f"▶️ Playing: {video_path}")

    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        continue

    playing = True
    fps = cap.get(cv2.CAP_PROP_FPS)
    skip_frames = int(fps * 30)  # 30 seconds forward

    while True:
        if playing:
            ret, frame = cap.read()
            if not ret:
                print("✅ Video finished.")
                break

            results = model(frame, conf=0.25)[0]

            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                avg_bgr = average_bgr(frame, (x1, y1, x2, y2))
                avg_rgb = (avg_bgr[2], avg_bgr[1], avg_bgr[0])

                # Draw detection
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
                cv2.circle(frame, (center_x, center_y), dot_radius, dot_color_bgr, -1)
                cv2.putText(frame, f"RGB{avg_rgb}", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2)

                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log.write(f"{os.path.basename(video_path)}, {timestamp}, RGB{avg_rgb}, Center({center_x},{center_y})\n")

            cv2.imshow("Filtered Patty Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            log.close()
            exit()
        elif key == ord('p'):
            playing = not playing
        elif key == 32:  # Spacebar to skip 30 seconds forward
            current = cap.get(cv2.CAP_PROP_POS_FRAMES)
            cap.set(cv2.CAP_PROP_POS_FRAMES, current + skip_frames)
        elif key in [13]:  # Enter to go to next video
            break

    cap.release()

cv2.destroyAllWindows()
log.close()
print("✅ All videos processed.")
