🖐️ Accessible Hand Gesture Controlled Virtual Mouse & Keyboard
Project Overview
This Python application transforms real-time hand gestures captured via webcam into precise mouse and keyboard commands for hands-free computer control. It features a robust multi-face authentication layer and utilizes computer vision for seamless navigation, scrolling, typing, and application switching across any active window (e.g., MS Word, web browser).

The project is built for high responsiveness by separating gesture detection from complex UI drawing logic, using native pyautogui commands for global input control. This intentional design choice eliminates the common lag associated with on-screen graphical overlays, ensuring a fluid and productive user experience, regardless of the active application. It bridges the gap between sophisticated computer vision capabilities and practical accessibility for daily tasks. The system is designed to be highly reliable, minimizing false inputs through calibrated distance thresholds and time-based delays for click events.

🌟 Features
Multi-Face Biometric Authentication: Secure access using the face_recognition library. The system dynamically loads and supports multiple authorized faces stored in the central reference_faces directory. This modular setup allows for easy management and expansion of authorized users without code modification.

Zero-Lag Global Input: Direct interaction with any active application (typing, clicking, scrolling) via pyautogui. Key presses and mouse actions are injected directly into the operating system's input stream, making the virtual controller fully functional even when the video feed is running in the background.

Intuitive Gesture Set (Left Hand - Mouse Control): The left hand is dedicated solely to navigation and interaction commands, making controls intuitive and consistent.

Cursor Movement: Hand position mapping using a clear Index finger extension gesture.

Left Click/Open: Index and Middle finger pinch (simulates double-click for direct file/folder execution).

Right Click: Thumb and Index finger pinch.

Scroll Up: Dedicated Thumb Up gesture (accelerated speed, currently set to 100 units per frame for quick navigation).

Scroll Down: All Fingers Down gesture (set to −50 units per frame).

Application Switch: Index and Pinky finger pinch (simulates the essential desktop shortcut: Alt+Tab).

Virtual Keyboard (Right Hand - Typing): The right hand serves as the typing interface, using a simple index finger pointer and an index/middle finger pinch to simulate physical key presses. This directly outputs characters to the focused application, providing a functional, lag-minimized typing experience.

🛠️ Installation and Setup
Prerequisites
You need Python 3.8+ installed. This project relies on several specialized libraries.

Clone the Repository:

git clone [YOUR_REPOSITORY_URL]
cd [YOUR_REPOSITORY_NAME]


Create and Activate Virtual Environment (Recommended):

python -m venv .venv
# On Windows:
.venv\Scripts\activate


Install Dependencies:

pip install opencv-python numpy pyautogui face-recognition mediapipe pyqt5 pynput


Face Authentication Setup
For the project to run, you must provide your reference images:

Create the Folder: Ensure a folder named reference_faces exists in the root directory of the project.

Add Images: Place one or more clear images of the authorized user's face (front view) inside the reference_faces folder. Supported formats: .jpg, .png, .jpeg.

▶️ How to Run
Ensure your webcam is available and not in use by other applications.

Run the main script:

python face_recognition_virtual_mouse.py


The webcam feed will open, prompting you to authenticate.

📦 Creating a Standalone Executable (.EXE)
To share this project as a simple application that runs on any Windows machine without requiring Python installation, use PyInstaller.

Install PyInstaller:

pip install pyinstaller


Build the Executable:
You must include the reference_faces folder and explicitly import cv2 (OpenCV). Run the following command from the project's root directory:

pyinstaller --onefile --add-data "reference_faces;reference_faces" --hidden-import "cv2" .venv\face_recognition_virtual_mouse.py


Distribution: The final .exe file will be in the newly created dist folder. To run the application on another PC, you must distribute the .exe file along with the empty reference_faces folder in the same directory.

🛠️ Development Notes
Lag Reduction: Complex UI drawing on the OpenCV frame was eliminated in favor of direct pyautogui calls, vastly improving frame rate and responsiveness.

Core Libraries: Uses MediaPipe for robust landmark detection and PyAutoGUI for OS-level input simulation.

Coordinate Scaling: All detected landmarks are scaled dynamically to accurately map hand positions to screen coordinates.
