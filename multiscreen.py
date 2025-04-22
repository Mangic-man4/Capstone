import tkinter as tk
from screeninfo import get_monitors
import cv2
from PIL import Image, ImageTk
import numpy as np
import torch
import subprocess
import threading
from ultralytics import YOLO
import os
import glob
from datetime import datetime

class CookingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Main Window")

        # Get all available monitors
        self.monitors = get_monitors()
        self.primary_monitor = self.monitors[0]
        self.secondary_monitor = self.monitors[1] if len(self.monitors) > 1 else self.primary_monitor 
        if len(self.monitors) == 1:
            print("Only one monitor detected! Running in single-screen mode.")
        
        # Set main window on the primary monitor
        self.root.geometry(f"{self.primary_monitor.width}x{self.primary_monitor.height}+{self.primary_monitor.x}+{self.primary_monitor.y}")
        self.root.state('zoomed')

        # Create the sidebar menu on the primary screen
        self.create_sidebar()

        # Track window resizing
        self.root.bind("<Configure>", self.update_sidebar_font)

        # Open Simulated View if a second monitor exists
        self.patties = []  # Store patties for simulation
        self.simulated_window = None
        if self.secondary_monitor:
            self.open_simulated_view()

        # Initialize webcam
        self.cap = None
        self.video_label = None

        # Load YOLO Model
        #self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', source='local') #Old
        self.model = YOLO("best.pt")  # model file name
        #yolo task=detect mode=predict model=yolov5s.pt source=0  <-- run in command prompt!!

    def find_camera(self):
        """Attempts to find an external camera, falls back to default webcam if not found."""
        """Only used with secondary camera find method"""
        for i in range(1, 2):  # Check for external cameras at indices 1, 2, 3, etc.
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                print(f"External camera found at index {i}.")
                return cap
            cap.release()
        print("No external camera found, falling back to default webcam at index 0.")
        cap = cv2.VideoCapture(0)  # Fallback to default webcam
        return cap

    
    def create_sidebar(self):
        """Creates a sidebar menu that adjusts based on window size."""
        self.sidebar = tk.Frame(self.root, bg="gray30", width=200)
        self.sidebar.pack(side="left", fill="y")

        self.menu_buttons = []
        menu_options = {
            "Main Menu": self.show_main_menu,
            "Settings": self.show_settings, # Disabled because tab is empty
            "Calibration": self.show_calibration, # Disabled because tab is empty
            "Coordinate Testing": self.show_coordinate_testing,
            "Webcam/Griddle View": self.show_griddle_view,
            "Simulated View": self.open_simulated_view,
            "AI Burger Webcam": self.show_burger_vision,
            "AI Burger Video": self.show_ai_burger_detection,
            "AI Burger UI": self.show_ai_burger_ui,
            "Exit": self.root.quit

        }

        for text, command in menu_options.items():
            #btn = tk.Button(self.sidebar, text=text, command=command, fg="white", bg="gray40")
            btn = tk.Button(self.sidebar, text=text, command=lambda cmd=command: self.switch_tab(cmd), fg="white", bg="gray40", disabledforeground="gray60")

            if text in ["Settings", "Calibration"]:
                btn.config(state="disabled")

            btn.pack(fill="x", pady=5)
            self.menu_buttons.append(btn)

        self.main_content = tk.Frame(self.root, bg="gray25")
        self.main_content.pack(side="right", fill="both", expand=True)

        self.current_screen = None
        self.show_main_menu()
    
    def switch_tab(self, command):
        """Switches tabs and ensures the webcam is properly released when needed."""
        if self.cap:
            self.close_webcam()
        command()

    def update_sidebar_font(self, event=None):
        """Dynamically adjusts sidebar button font size based on window height."""
        window_height = self.root.winfo_height()
        font_size = max(10, window_height // 55)  # Adjust based on screen size

        for button in self.menu_buttons:
            button.config(font=("Arial", font_size))

    def switch_screen(self, text):
        if self.current_screen:
            self.current_screen.destroy()
        self.current_screen = tk.Frame(self.main_content, bg="gray25")
        self.current_screen.pack(fill="both", expand=True)
        label = tk.Label(self.current_screen, text=text, font=("Arial", 18), fg="white", bg="gray25")
        label.pack(pady=100)


    def show_main_menu(self): self.switch_screen("🏠 Main Menu")
    def show_settings(self): self.switch_screen("⚙️ Settings")
    def show_calibration(self): self.switch_screen("🔧 Calibration")
    def show_coordinate_testing(self): self.setup_coordinate_testing()

    def show_griddle_view(self):
        """Displays the live video feed and applies color-based burger detection."""
        self.switch_screen("📷 Webcam/Griddle View")
        self.video_label = tk.Label(self.current_screen)
        self.video_label.pack()
        
        #self.cap = self.find_camera()  # Use the method to find the correct camera (secondary method)
        #self.cap = cv2.VideoCapture(1)  # Default method (0 for pc's webcam, 1 for external camera)
        self.cap = self.find_camera()


        #if not self.cap.isOpened():  # Check if the camera opened successfully
        #    print("External camera not found. Falling back to default webcam at index 0.")
        #   self.cap = cv2.VideoCapture(0)  # Fallback to the built-in webcam (index 0)

        self.update_webcam_feed()
        

    def show_burger_vision(self): 
        self.switch_screen("🍔 YOLOv8 Webcam RGB Tracker")

        """def launch_external_yolo_webcam():
            subprocess.Popen(["python", "RGB_Tracker_Webcam.py"]) # Launch the external script in same directory

        threading.Thread(target=launch_external_yolo_webcam, daemon=True).start()

        self.video_label = tk.Label(self.current_screen, text="Launching AI Burger Webcam...")
        self.video_label.pack()"""

        threading.Thread(target=self.webcam_yolo_v8, daemon=True).start()

        """Displays Burger Vision analysis in the main window."""

        #self.switch_screen("🍔 Burger Vision Analysis")
        
        self.image_label = tk.Label(self.current_screen, bg="gray25")
        self.image_label.pack()
        
        self.analysis_label = tk.Label(self.current_screen, text="", font=("Arial", 14), fg="white", bg="gray25")
        self.analysis_label.pack(pady=10)

        self.color_display = tk.Canvas(self.current_screen, width=300, height=50, bg="gray25", highlightthickness=0)
        self.color_display.pack(pady=5)
        
        #self.analyze_burger_images()
        #self.update_burger_image()

    def show_ai_burger_ui(self):
        """Displays the AI Burger UI in the main window."""
        self.switch_screen("AI Burger UI")

        # Call the original YOLO_UI script as a separate process
        def launch_external_yolo():
            subprocess.Popen(["python", "YOLO_UI.py"]) # Launch the external script in same directory

        threading.Thread(target=launch_external_yolo, daemon=True).start()

        self.video_label = tk.Label(self.current_screen, font=("Arial", 12), fg="white", bg="gray25", text="Launching YOLO_UI.py...")
        self.video_label.pack(pady=200)

        

    def update_webcam_feed(self):
        """Captures video frames, applies burger detection, and updates the Tkinter UI."""
        if self.cap:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)  # Flip horizontally for natural mirroring
                processed_frame = self.detect_burgers(frame)  # Apply burger detection
                img = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                img = ImageTk.PhotoImage(img)
                
                self.video_label.config(image=img)
                self.video_label.image = img
                
            self.current_screen.after(10, self.update_webcam_feed)  # Refresh at ~60 FPS

    def detect_burgers(self, frame):
        """Detects burger patties using color and shape analysis."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define color range for detecting browned patties (adjust if needed)
        lower_brown = np.array([5, 50, 50])  # Lower HSV threshold
        upper_brown = np.array([30, 255, 255])  # Upper HSV threshold
        mask = cv2.inRange(hsv, lower_brown, upper_brown)
        
        # Find contours from the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 500:  # Ignore small noise
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Draw bounding box
                cv2.putText(frame, "Burger", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame

    def close_webcam(self):
        """Releases the webcam when switching away from the Webcam/Griddle or AI Burger Detection View."""
        if self.cap:
            self.cap.release()
            self.cap = None

    def show_ai_burger_detection(self):
        self.switch_screen("📼 YOLOv8 Video RGB Tracker")
        """Displays AI-based burger detection in the main window."""

        """def launch_external_yolo_video():
            subprocess.Popen(["python", "RGB_Tracker_Video.py"]) # Launch the external script in same directory

        threading.Thread(target=launch_external_yolo_video, daemon=True).start()

        self.video_label = tk.Label(self.current_screen, text="Launching AI Burger Video...")
        self.video_label.pack()"""
        
        threading.Thread(target=self.video_yolo_v8, daemon=True).start()
        #self.switch_screen("🤖 AI Burger Detection")
        self.video_label = tk.Label(self.current_screen)
        self.video_label.pack()
        
        """self.cap = cv2.VideoCapture(1)  # External camera
        if not self.cap.isOpened():
            print("External camera not found. Falling back to default webcam.")
            self.cap = cv2.VideoCapture(0)
        
        self.update_ai_detection()"""

    def update_ai_detection(self): # Don't use beacuse it uses YOLOv5
        """Runs YOLOv5 on the live feed and updates the UI."""
        if self.cap:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                processed_frame = self.detect_ai_burgers(frame)
                img = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                img = ImageTk.PhotoImage(img)
                
                self.video_label.config(image=img)
                self.video_label.image = img
            
            self.current_screen.after(10, self.update_ai_detection)
    
    def detect_ai_burgers(self, frame):
        """Uses YOLOv5 to detect burgers and their doneness levels."""
        results = self.model(frame)
        detections = results.pandas().xyxy[0]
        
        for _, row in detections.iterrows():
            x1, y1, x2, y2, label, conf = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax']), row['name'], row['confidence']
            color = (0, 255, 0) if label == 'raw' else (0, 255, 255) if label == 'flip_ready' else (255, 0, 0) if label == 'flipped' else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame

    def webcam_yolo_v8(self):
        model = YOLO("best.pt")
        cap = self.find_camera()
        dot_radius = 8
        dot_color_rgb = (0, 0, 255)
        dot_color_bgr = (dot_color_rgb[2], dot_color_rgb[1], dot_color_rgb[0])

        def average_bgr(image, box):
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
                avg_bgr = average_bgr(frame, (x1, y1, x2, y2))
                avg_rgb = (avg_bgr[2], avg_bgr[1], avg_bgr[0])
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
                cv2.circle(frame, (center_x, center_y), dot_radius, dot_color_bgr, -1)
                text = f"RGB{avg_rgb}"
                cv2.putText(frame, text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow("YOLOv8 Webcam RGB Tracker", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def video_yolo_v8(self):
        model = YOLO("best.pt")
        video_folder = "Burger_video"
        dot_radius = 4
        dot_color_rgb = (255, 0, 0)
        dot_color_bgr = (dot_color_rgb[2], dot_color_rgb[1], dot_color_rgb[0])

        def average_bgr(image, box):
            x1, y1, x2, y2 = map(int, box)
            roi = image[y1:y2, x1:x2]
            if roi.size == 0:
                return (0, 0, 0)
            mean_bgr = np.mean(roi.astype(np.float32), axis=(0, 1))
            return tuple(map(int, mean_bgr))

        video_paths = sorted(glob.glob(os.path.join(video_folder, "*.mp4")))
        if not video_paths:
            print(f"❌ No videos found in folder: {video_folder}")
            return

        log_file = "patty_rgb_log.txt"
        log = open(log_file, "w")

        for video_path in video_paths:
            cap = cv2.VideoCapture(video_path)
            print(f"▶️ Playing: {video_path}")
            if not cap.isOpened():
                print(f"❌ Could not open video: {video_path}")
                continue

            playing = True
            fps = cap.get(cv2.CAP_PROP_FPS)
            skip_frames = int(fps * 30)

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
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
                        cv2.circle(frame, (center_x, center_y), dot_radius, dot_color_bgr, -1)
                        cv2.putText(frame, f"RGB{avg_rgb}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        log.write(f"{os.path.basename(video_path)}, {timestamp}, RGB{avg_rgb}, Center({center_x},{center_y})\\n")

                    cv2.imshow("Filtered Patty Detection", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    log.close()
                    return
                elif key == ord('p'):
                    playing = not playing
                elif key == 32:
                    current = cap.get(cv2.CAP_PROP_POS_FRAMES)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, current + skip_frames)
                elif key == 13:
                    break

            cap.release()

        cv2.destroyAllWindows()
        log.close()
        print("✅ All videos processed.")

    def ui_yolo_v8(self):
        # Deprecated: Using external YOLO_UI.py script via subprocess
        pass
        

    def analyze_burger_images(self):
        """Loads burger images and dynamically determines cooking states."""
        try:
            self.raw_patty = cv2.imread("patty_raw.png")
            self.half_patty = cv2.imread("patty_half_cooked.png")
            self.cooked_patty = cv2.imread("patty_ready.png")

            if self.raw_patty is None or self.half_patty is None or self.cooked_patty is None:
                print("Error: One or more images not found!")
                self.analysis_label.config(text="Error: One or more images not found!")
                return

            print("All images loaded successfully!")
            height_raw, width_raw, _ = self.raw_patty.shape
            self.half_patty = cv2.resize(self.half_patty, (width_raw, height_raw))
            self.cooked_patty = cv2.resize(self.cooked_patty, (width_raw, height_raw))

            # Convert to HSV
            hsv_raw = cv2.cvtColor(self.raw_patty, cv2.COLOR_BGR2HSV)
            hsv_half = cv2.cvtColor(self.half_patty, cv2.COLOR_BGR2HSV)
            hsv_cooked = cv2.cvtColor(self.cooked_patty, cv2.COLOR_BGR2HSV)

            # Extract dominant hue values using the center region of the patty
            hues = [
                self.get_dominant_hue(hsv_raw),
                self.get_dominant_hue(hsv_half),
                self.get_dominant_hue(hsv_cooked)
            ]

            # Assign fixed categories
            labels = ["Raw (10%)", "Half-Cooked (60%)", "Fully Cooked (100%)"]
            colors = [self.hue_to_rgb(hue) for hue in hues]
            
            result_text = "\nCooking State Detection:\n"
            self.color_display.delete("all")
            for i, (hue, label, color) in enumerate(zip(hues, labels, colors)):
                result_text += f" {label}: Hue {hue}\n"
                self.color_display.create_rectangle(10 + i * 100, 10, 90 + i * 100, 40, fill=color, outline="white")
            
            self.analysis_label.config(text=result_text)
        
        except Exception as e:
            print(f"Error in processing images: {e}")
            self.analysis_label.config(text=f"Error: {e}")

    def get_dominant_hue(self, hsv_image):
        """Finds the dominant hue value in the central region of an image, ignoring dark pixels."""
        h, w, _ = hsv_image.shape
        center_region = hsv_image[h//4:3*h//4, w//4:3*w//4, 0]  # Crop center
        hue_values = center_region.flatten()
        hue_values = hue_values[hue_values > 10]  # Ignore dark pixels
        if len(hue_values) == 0:
            return 0
        return int(np.median(hue_values))
    
    def hue_to_rgb(self, hue):
        """Converts a hue value to an approximate RGB color."""
        import colorsys
        r, g, b = colorsys.hsv_to_rgb(hue / 180.0, 1, 1)
        return f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'

    def update_burger_image(self):
        """Displays the combined burger cooking images in the UI."""
        combined_image = np.hstack((self.raw_patty, self.half_patty, self.cooked_patty))
        combined_image = cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(combined_image)
        img = img.resize((600, 200))
        img_tk = ImageTk.PhotoImage(img)

        self.image_label.configure(image=img_tk)
        self.image_label.image = img_tk

    def setup_coordinate_testing(self):
        """Creates UI for entering patty positions and timer duration."""
        if self.current_screen:
            self.current_screen.destroy()
        self.current_screen = tk.Frame(self.main_content, bg="gray25") #-----------------
        self.current_screen.pack(fill="both", expand=True)

        frame = tk.Frame(self.current_screen, bg="gray25")
        frame.pack(pady=400)

        tk.Label(frame, text="X Position:", bg="gray25", fg="white", font=("Arial", 15)).grid(row=0, column=0)
        tk.Label(frame, text="Y Position:", bg="gray25", fg="white", font=("Arial", 15)).grid(row=1, column=0)
        tk.Label(frame, text="Timer (seconds):", bg="gray25", fg="white", font=("Arial", 15)).grid(row=2, column=0)

        self.x_entry = tk.Entry(frame)
        self.y_entry = tk.Entry(frame)
        self.time_entry = tk.Entry(frame)
        self.x_entry.grid(row=0, column=1)
        self.y_entry.grid(row=1, column=1)
        self.time_entry.grid(row=2, column=1)

        add_button = tk.Button(frame, text="Add Patty", command=self.add_patty)
        add_button.grid(row=3, column=0, columnspan=2, pady=10)

    def open_simulated_view(self):
        """Creates or reopens the secondary window for displaying simulated patties."""
        if self.simulated_window and tk.Toplevel.winfo_exists(self.simulated_window):
            self.simulated_window.deiconify()
            return
        
        self.simulated_window = tk.Toplevel(self.root)
        self.simulated_window.title("Simulated View")
        self.simulated_window.geometry(f"{self.secondary_monitor.width}x{self.secondary_monitor.height}+{self.secondary_monitor.x}+{self.secondary_monitor.y}")
        self.simulated_window.state('zoomed')
        self.simulated_window.protocol("WM_DELETE_WINDOW", self.hide_simulated_view)

        self.canvas = tk.Canvas(self.simulated_window, bg="black")
        self.canvas.pack(fill="both", expand=True)

    def hide_simulated_view(self):
        """Hides the simulated view instead of destroying it."""
        self.simulated_window.withdraw()

    def add_patty(self):
        """Adds a patty with a countdown timer."""
        try:
            x = int(self.x_entry.get())
            y = int(self.y_entry.get())
            time_value = int(self.time_entry.get()) if self.time_entry.get() else 30
        except ValueError:
            print("Invalid input! Enter integer values for X, Y, and Time.")
            return

        patty = {'x': x, 'y': y, 'time': time_value, 'circle': None, 'text': None, 'blinking': False, 'blink_state': True}
        self.patties.append(patty)
        self.update_simulated_view()
        self.start_timer(patty)

    def update_simulated_view(self):
        """Redraws patties on the simulated screen."""
        if not hasattr(self, 'canvas'):
            return
        self.canvas.delete("all")
        
        for patty in self.patties:
            color = "green" if patty['time'] > 10 else "orange" if patty['time'] > 5 else "red"
            patty['circle'] = self.canvas.create_oval(
                patty['x'] - 150, patty['y'] - 150,
                patty['x'] + 150, patty['y'] + 150,
                fill=color
            )
            patty['text'] = self.canvas.create_text(
                patty['x'], patty['y'], 
                text=str(patty['time']), 
                font=("Arial", 30), 
                fill="white" if patty ['time'] > 10 else "black" if patty['time'] > 5 else "white"
            )
    
    def start_timer(self, patty):
        """Countdown for the patty timer using after()."""
        def countdown():
            if patty['time'] >= 0:
                self.update_simulated_view()
                patty['time'] -= 1

                if patty['time'] == 4 and not patty['blinking']:
                    patty['blinking'] = True
                    self.blink_patty(patty)

                self.root.after(1000, countdown)
            else:
                self.patties.remove(patty)
                self.update_simulated_view()
        countdown()

    def blink_patty(self, patty):
        """Blinks the patty at a constant rate of 0.4 seconds."""
        def toggle():
            if patty in self.patties and patty['time'] >= 0:
                patty['blink_state'] = not patty['blink_state']
                new_color = "red" if patty['blink_state'] else "black"
                self.canvas.itemconfig(patty['circle'], fill=new_color)
                self.root.after(400, toggle)

        if patty['blinking']:
            toggle()

if __name__ == "__main__":
    root = tk.Tk()
    app = CookingApp(root)
    root.mainloop()
