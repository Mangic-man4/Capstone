import tkinter as tk
from screeninfo import get_monitors

class CookingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Main Window")

        # Get all available monitors
        self.monitors = get_monitors()

        if len(self.monitors) < 2:
            print("Only one monitor detected! Running in single-screen mode.")
            self.root.state('zoomed')
            return

        self.primary_monitor = self.monitors[0]  # Primary screen
        self.secondary_monitor = self.monitors[1]  # Secondary screen

        # Set main window on the primary monitor
        print(f"Placing primary window at: {self.primary_monitor.x}, {self.primary_monitor.y}")
        self.root.geometry(f"{self.primary_monitor.width}x{self.primary_monitor.height}+{self.primary_monitor.x}+{self.primary_monitor.y}")
        self.root.state('zoomed')

        # Create the sidebar menu on the primary screen
        self.create_sidebar()

        # Create second window
        self.open_secondary_window()

        # Track window resizing
        self.root.bind("<Configure>", self.update_sidebar_font)

    def create_sidebar(self):
        """Creates a sidebar menu that adjusts based on window size."""
        self.sidebar = tk.Frame(self.root, bg="gray30", width=200)
        self.sidebar.pack(side="left", fill="y")

        self.menu_buttons = []  # Store buttons for font resizing

        menu_options = {
            "Main Menu": self.show_main_menu,
            "Camera Views": self.show_camera,
            "Griddle View": self.show_griddle,
            "Calibration": self.show_calibration,
            "Settings": self.show_settings
        }

        for text, command in menu_options.items():
            btn = tk.Button(self.sidebar, text=text, command=command, fg="white", bg="gray40")
            btn.pack(fill="x", pady=5)  # Add padding between buttons
            self.menu_buttons.append(btn)  # Store button reference

        self.main_content = tk.Frame(self.root, bg="gray25")
        self.main_content.pack(side="right", fill="both", expand=True)

        self.current_screen = None
        self.show_main_menu()

    def update_sidebar_font(self, event=None):
        """Dynamically adjusts sidebar button font size based on window height."""
        window_height = self.root.winfo_height()
        font_size = max(10, window_height // 40)  # Adjust based on screen size

        for button in self.menu_buttons:
            button.config(font=("Arial", font_size))

    def switch_screen(self, text):
        """Updates the primary screen content dynamically."""
        if self.current_screen:
            self.current_screen.destroy()
        self.current_screen = tk.Label(self.main_content, text=text, font=("Arial", 18), fg="white", bg="gray25")
        self.current_screen.pack(expand=True)

    def show_main_menu(self): self.switch_screen("🏠 Main Menu")
    def show_camera(self): self.switch_screen("📷 Camera Views")
    def show_griddle(self): self.switch_screen("🔥 Griddle View")
    def show_calibration(self): self.switch_screen("⚙️ Calibration")
    def show_settings(self): self.switch_screen("⚙️ Settings")

    def open_secondary_window(self):
        """Opens the second window on the secondary monitor."""
        self.second_window = tk.Toplevel(self.root)
        self.second_window.title("Secondary Window")

        print(f"Placing secondary window at: {self.secondary_monitor.x}, {self.secondary_monitor.y}")
        self.second_window.geometry(f"{self.secondary_monitor.width}x{self.secondary_monitor.height}+{self.secondary_monitor.x}+{self.secondary_monitor.y}")
        self.second_window.state('zoomed')

        # Label in the secondary window
        self.label_on_second_window = tk.Label(self.second_window, text="🔥 Griddle Display", font=("Arial", 24))
        self.label_on_second_window.pack(pady=50)

        # Example dynamic update button
        update_button = tk.Button(self.second_window, text="Update Griddle View", command=self.update_text)
        update_button.pack(pady=20)

    def update_text(self):
        """Updates the secondary screen dynamically."""
        self.label_on_second_window.config(text="Updated Griddle Data!")

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = CookingApp(root)
    root.mainloop()
