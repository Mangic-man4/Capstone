import tkinter as tk
from screeninfo import get_monitors

def update_text():
    # Change the text dynamically on the second window
    label_on_second_window.config(text="Text updated dynamically!")

def open_windows():
    # Get all available monitors
    monitors = get_monitors()

    if len(monitors) < 2:
        print("Only one monitor detected! Running in single-screen mode.")
        root.state('zoomed')  # Maximize window if only one monitor
        return

    primary_monitor = monitors[0]  # Assume first monitor is primary
    secondary_monitor = monitors[1]  # Assume second monitor is secondary

    # Set main window on the primary monitor
    print(f"Placing primary window at: {primary_monitor.x}, {primary_monitor.y}")
    root.geometry(f"{primary_monitor.width}x{primary_monitor.height}+{primary_monitor.x}+{primary_monitor.y}")
    root.update()  # Force update to apply geometry first

    # Maximize the main window (root) on the primary monitor
    root.state('zoomed')

    # Open the second window on the secondary monitor
    second_window = tk.Toplevel(root)
    second_window.title("Secondary Window")
    print(f"Placing secondary window at: {secondary_monitor.x}, {secondary_monitor.y}")
    second_window.geometry(f"{secondary_monitor.width}x{secondary_monitor.height}+{secondary_monitor.x}+{secondary_monitor.y}")
    second_window.update()  # Force update to apply geometry first

    # Maximize the second window (on the secondary monitor)
    second_window.state('zoomed')

    # Label in the secondary window
    global label_on_second_window  # We need a reference to this label to update it later
    label_on_second_window = tk.Label(second_window, text="This is the secondary screen", font=("Arial", 24))
    label_on_second_window.pack(pady=50)

    # Add a button to update the text dynamically
    update_button = tk.Button(second_window, text="Update Text", command=update_text)
    update_button.pack(pady=20)

# Create the main window
root = tk.Tk()
root.title("Main Window")

# Label for the main menu
label = tk.Label(root, text="Main Menu", font=("Arial", 32))
label.pack(pady=50)

# Button to start maximizing the windows on both screens
button = tk.Button(root, text="Open Maximized on Both Screens", command=open_windows)
button.pack(pady=20)

# Run the application
root.mainloop()
