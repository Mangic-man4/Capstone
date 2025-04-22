import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")

# Start webcam capture
cap = cv2.VideoCapture(0)

# === Configurable parameters ===
dot_radius = 8  # Size of the center dot
dot_color_rgb = (0, 0, 255)  # Red dot in RGB format

# Convert RGB to BGR for OpenCV drawing
dot_color_bgr = (dot_color_rgb[2], dot_color_rgb[1], dot_color_rgb[0])

def average_bgr(image, box):
    """
    Returns the average BGR color of the region of interest in 'box'.
    """
    x1, y1, x2, y2 = map(int, box)
    roi = image[y1:y2, x1:x2]

    roi_float = roi.astype(np.float32)
    mean_bgr = np.mean(roi_float, axis=(0, 1))
    return tuple(map(int, mean_bgr))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

        # Average color in ROI
        avg_bgr = average_bgr(frame, (x1, y1, x2, y2))
        avg_rgb = (avg_bgr[2], avg_bgr[1], avg_bgr[0])

        # Draw bounding box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        # Draw center dot
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        cv2.circle(frame, (center_x, center_y), dot_radius, dot_color_bgr, -1)

        # Draw text with RGB value
        text = f"RGB{avg_rgb}"
        cv2.putText(
            frame,
            text,
            (int(x1), int(y1) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    cv2.imshow("Multiple Object RGB Tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
