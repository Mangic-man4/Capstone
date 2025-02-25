import tkinter as tk
from screeninfo import get_monitors

class CookingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Testing Console")
        
        # Get monitors
        self.monitors = get_monitors()
        self.primary_monitor = self.monitors[0]
        self.secondary_monitor = self.monitors[1] if len(self.monitors) > 1 else None
        
        # Set window position
        self.root.geometry(f"{self.primary_monitor.width}x{self.primary_monitor.height}+{self.primary_monitor.x}+{self.primary_monitor.y}")
        self.root.state('zoomed')
        
        # UI for Testing Console
        self.setup_testing_console()
        
        # Open Secondary Window (Testing View)
        self.open_testing_view()
        
        # Store patties data
        self.patties = []
    
    def setup_testing_console(self):
        """Creates UI for entering patty positions and timer duration"""
        frame = tk.Frame(self.root)
        frame.pack(pady=20)
        
        tk.Label(frame, text="X Position:").grid(row=0, column=0)
        tk.Label(frame, text="Y Position:").grid(row=1, column=0)
        tk.Label(frame, text="Timer (seconds):").grid(row=2, column=0)
        
        self.x_entry = tk.Entry(frame)
        self.y_entry = tk.Entry(frame)
        self.time_entry = tk.Entry(frame)
        self.x_entry.grid(row=0, column=1)
        self.y_entry.grid(row=1, column=1)
        self.time_entry.grid(row=2, column=1)
        
        add_button = tk.Button(frame, text="Add Patty", command=self.add_patty)
        add_button.grid(row=3, column=0, columnspan=2, pady=10)
    
    def open_testing_view(self):
        """Creates the secondary window for displaying patties"""
        if not self.secondary_monitor:
            print("Only one monitor detected! Secondary view not available.")
            # Use a placeholder Label when no secondary monitor is detected
            self.canvas = tk.Label(self.root, text="No second monitor detected.", bg="gray", font=("Arial", 20))
            self.canvas.pack(fill="both", expand=True)
            return
        
        self.second_window = tk.Toplevel(self.root)
        self.second_window.title("Testing View")
        self.second_window.geometry(f"{self.secondary_monitor.width}x{self.secondary_monitor.height}+{self.secondary_monitor.x}+{self.secondary_monitor.y}")
        self.second_window.state('zoomed')
        
        self.canvas = tk.Canvas(self.second_window, bg="black")
        self.canvas.pack(fill="both", expand=True)
    
    def add_patty(self):
        """Adds a patty with a countdown timer"""
        try:
            x = int(self.x_entry.get())
            y = int(self.y_entry.get())
            time_value = int(self.time_entry.get()) if self.time_entry.get() else 30
        except ValueError:
            print("Invalid input! Enter integer values for X, Y, and Time.")
            return
        
        patty = {'x': x, 'y': y, 'time': time_value, 'circle': None, 'text': None, 'blinking': False, 'blink_state': True}
        self.patties.append(patty)
        
        # Start countdown using after method
        self.update_testing_view()
        self.start_timer(patty)
    
    def update_testing_view(self):
        """Redraws patties on the secondary screen"""
        if not hasattr(self, 'canvas'):
            return
        
        if isinstance(self.canvas, tk.Label):  # If it's a Label (no second monitor)
            return  # Do nothing
        
        # Clear canvas if it's a Canvas widget
        self.canvas.delete("all")
        
        for patty in self.patties:
            color = "green" if patty['time'] > 10 else "orange" if patty['time'] > 5 else "red"
            patty['circle'] = self.canvas.create_oval(
                patty['x'] - 150, patty['y'] -  150,
                patty['x'] + 150, patty['y'] + 150,
                fill=color
            )
            patty['text'] = self.canvas.create_text(
                patty['x'], patty['y'] - 5, 
                text=str(patty['time']), 
                font=("Arial", 30), 
                fill="white" if patty ['time'] > 10 else "black" if patty['time'] > 5 else "white"
            )
    
    def start_timer(self, patty):
        """Countdown for the patty timer using after()"""
        def countdown():
            if patty['time'] > 0:
                patty['time'] -= 1
                self.update_testing_view()
                
                # Start blinking when time <= 10
                if patty['time'] == 5 and not patty['blinking']:
                    patty['blinking'] = True
                    self.blink_patty(patty)
                
                self.root.after(1000, countdown)  # Update every second
            else:
                # Remove patty when time is up
                self.patties.remove(patty)
                self.update_testing_view()
        
        countdown()
    
    def blink_patty(self, patty):
        """Blinks the patty at a constant rate of 0.4 seconds using Tkinter's after()"""
        def toggle():
            if patty in self.patties and patty['time'] > 0:
                patty['blink_state'] = not patty['blink_state']
                new_color = "red" if patty['blink_state'] else "black"
                self.canvas.itemconfig(patty['circle'], fill=new_color)
                self.root.after(400, toggle)  #  0.4s interval blinking
        
        # Start blinking only if it's not already blinking
        if not patty['blinking']:
            patty['blinking'] = True
            toggle()
        #if patty['blinking']:
        #    toggle()

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = CookingApp(root)
    root.mainloop()
