import cv2
import numpy as np

# Load images
raw_patty = cv2.imread("patty_raw.png")
half_patty = cv2.imread("patty_half_cooked.png")
cooked_patty = cv2.imread("patty_ready.png")

if raw_patty is None or half_patty is None or cooked_patty is None:
    print("Error: One or more images not found!")
    exit()

print("All images loaded successfully!")

# Resize images to match dimensions
height_raw, width_raw, _ = raw_patty.shape
half_patty_resized = cv2.resize(half_patty, (width_raw, height_raw))
cooked_patty_resized = cv2.resize(cooked_patty, (width_raw, height_raw))

# Convert images to HSV
hsv_raw = cv2.cvtColor(raw_patty, cv2.COLOR_BGR2HSV)
hsv_half = cv2.cvtColor(half_patty_resized, cv2.COLOR_BGR2HSV)
hsv_cooked = cv2.cvtColor(cooked_patty_resized, cv2.COLOR_BGR2HSV)

# Extract the average HSV values
avg_raw = np.mean(hsv_raw, axis=(0, 1))
avg_half = np.mean(hsv_half, axis=(0, 1))
avg_cooked = np.mean(hsv_cooked, axis=(0, 1))

print(f"\nRaw Patty (HSV): {avg_raw}")
print(f"Half-Cooked Patty (HSV): {avg_half}")
print(f"Cooked Patty (HSV): {avg_cooked}")

# Get hue values
raw_hue, raw_sat, raw_val = avg_raw
half_hue, half_sat, half_val = avg_half
cooked_hue, cooked_sat, cooked_val = avg_cooked

print("\nColor Analysis:")

# **Improved Hue Classification with Cooking Progress at Each Step**
if 0 <= raw_hue <= 30 or 160 <= raw_hue <= 180:  
    print("✅ The raw patty is properly red (uncooked). 10%")

if 30 < half_hue < 50:  
    print("✅ Half-cooked patty is transitioning in color (partially cooked). 60%")

if 5 <= cooked_hue <= 25:  
    print("✅ Cooked patty is properly browned. 100%")

# Display the images side by side
combined_image = np.hstack((raw_patty, half_patty_resized, cooked_patty_resized))

# Add text labels
font = cv2.FONT_HERSHEY_SIMPLEX
y_position = height_raw - 10
cv2.putText(combined_image, 'Raw 10%', (50, y_position), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
cv2.putText(combined_image, 'Half-Cooked 60%', (width_raw + 50, y_position), font, 1, (0, 255, 255), 2, cv2.LINE_AA)
cv2.putText(combined_image, 'Cooked 100%', (2 * width_raw + 50, y_position), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

# Show the combined image
cv2.imshow("Patty Cooking States", combined_image)

# **Fix for Laggy Window Closing**
while True:
    key = cv2.waitKey(1) & 0xFF
    if key in [27, 8, ord('q')]:  # ESC (27), Backspace (8), or 'q'
        break

cv2.destroyAllWindows()
