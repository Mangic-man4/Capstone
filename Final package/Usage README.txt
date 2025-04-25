What is this software?
This application is a is an interactive cooking assistant that uses computer vision to detect burgers on screen. 
It provides a real-time video feed and helps you track and manage your cooking process by detecting items like burgers automatically.



What does it do?
Real-time Object Detection: The app detects and highlights burgers in a live video feed using the YOLOv8 model.

Simple User Interface: The interface is straightforward, with buttons that allow you to switch between different sections (tabs) of the app.

Manual Multi-Screen Support: While the app supports multiple screens, you can manually drag the app window to any monitor you wish to use.
Live Feed or Video Detection:
    AI Burger Webcam: Displays live feed from a connected camera and detects burgers in real-time.
    AI Burger Video: Shows live detection from a video file, detecting burgers in real-time.
    AI Burger UI: Similar to the video tab, but tailored for UI-based tasks, providing live detection from videos.

Note: The following tabs were used for development and are either non-functional or not necessary for the end user:
    Coordinate Testing
    Simulated View
    Webcam/Griddle View



How to Use the Application?
1. Start the Application:
    After following the installation steps, open the terminal/command prompt and run the application by typing:
        python multiscreen.py



2. Select Your Monitor:
    The app supports multiple screens, but you need to manually drag the application window to the monitor you wish to use. 
    Once the window is on your desired screen, the app will function as expected.



3. Switch Between Tabs:
The app provides tabs for different functionalities:

    AI Burger Webcam: Displays a live feed from a connected camera and detects burgers in real-time.
        Press 'q' to stop the live feed and exit the webcam view.

    AI Burger Video: Shows live detection from a video file and detects burgers in real-time.
        Press 'q' or Enter to stop the video feed.
        Press 'p' to toggle play/pause for the video.
        Press Spacebar to skip forward 5 seconds in the video.

    AI Burger UI: Similar to the video tab, it provides live detection from video files but with a user interface tailored for specific tasks.
        Press Esc to exit the live detection and stop the process.
        Press Spacebar to skip forward 5 seconds in the video feed.

    Coordinate Testing: Allows you to insert x and y coordinates and a time, where a "simulated" patty will appear on the 'Simulated View' screen based on the input coordinates.

    Simulated View: Displays the simulated patties from 'Coordinate Testing'. 

    Webcam/Griddle View: This tab was a test for the webcam view and YOLO integration but is currently not functioning properly.



4. Start Object Detection:
    Click on the respective tab to start detecting burgers in real-time (either from a live feed or a video).



5. Stop Detection:
    To stop the detection process, close the application or press the correct button ('q', Enter or Esc depending on the tab).




