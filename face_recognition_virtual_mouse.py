import cv2
import face_recognition
import numpy as np
import pyautogui
import time
import math
import mediapipe as mp
import os
from PyQt5 import QtWidgets, QtCore
from pynput.keyboard import Controller
from time import sleep

attempt_counter = 3  # Initialize the counter with 3 attempts

class Button:
    def __init__(self, pos, text, size=[85, 85]):
        self.pos = pos
        self.text = text
        self.size = size

def face_recognition_auth():
    global attempt_counter

    # Create the reference folder if it doesn't exist
    ref_folder = "reference_faces"
    if not os.path.exists(ref_folder):
        os.makedirs(ref_folder)
        print(f"Created folder: {ref_folder}. Please place your reference images inside.")
        exit()

    # Get a list of all image files in the reference folder
    ref_images_paths = [os.path.join(ref_folder, f) for f in os.listdir(ref_folder)
                        if f.endswith(('.jpg', '.png', '.jpeg'))]

    if not ref_images_paths:
        print(f"No reference images found in the '{ref_folder}' folder.")
        exit()

    known_face_encodings = []
    print(f"Loading {len(ref_images_paths)} reference images...")

    for image_path in ref_images_paths:
        try:
            ref_image = cv2.imread(image_path)
            ref_image_rgb = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
            
            # Find all faces in the reference image
            face_encodings = face_recognition.face_encodings(ref_image_rgb)
            if len(face_encodings) > 0:
                known_face_encodings.extend(face_encodings)
                print(f"✅ Faces loaded from: {image_path}")
            else:
                print(f"❌ No face detected in: {image_path}")
        except Exception as e:
            print(f"❌ Error loading {image_path}: {e}")

    if not known_face_encodings:
        print("No valid face encodings found in the reference images. Please try again with clear photos.")
        exit()
    
    print(f"✅ Loaded {len(known_face_encodings)} face encodings from the reference folder.")

    while attempt_counter > 0:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        # Give the user 5 seconds to get ready
        start_time = time.time()
        while time.time() - start_time < 5:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            cv2.putText(frame, "Get ready...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Webcam Feed", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Process the frame after the "Get ready" period
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            cap.release()
            cv2.destroyAllWindows()
            return False

        # Resize the frame to reduce processing time (optional, tweak for performance)
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        # Convert the frame to RGB for face recognition
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Find all faces in the frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        if len(face_encodings) == 0:
            # No face detected, display message and exit after 5 seconds
            cv2.putText(frame, "No face detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f"Attempts left: {attempt_counter - 1}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Webcam Feed", frame)
            cv2.waitKey(5000)  # Wait for 5 seconds
            cap.release()
            cv2.destroyAllWindows()

            attempt_counter -= 1
            if attempt_counter == 0:
                show_too_many_attempts_dialog()
                exit()
            continue

        match_found = False  # Flag to check if any match is found

        # Compare all detected faces with the reference face
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            match = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
            
            # Draw a rectangle around each face
            cv2.rectangle(frame, (left*2, top*2), (right*2, bottom*2), (0, 255, 0), 2)  # Rescale coordinates
            
            if True in match:
                match_found = True
                cv2.putText(frame, "Match", (left*2, top*2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "No Match", (left*2, top*2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if match_found:
            # Display "Access Granted" and exit after 5 seconds
            cv2.putText(frame, "Access Granted", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Webcam Feed", frame)
            cv2.waitKey(5000)  # Wait for 5 seconds
            cap.release()
            cv2.destroyAllWindows()
            return True
        else:
            # Display "Access Denied" and show the retry GUI
            attempt_counter -= 1
            cv2.putText(frame, "Access Denied", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f"Attempts left: {attempt_counter}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Webcam Feed", frame)
            cv2.waitKey(5000)  # Wait for 5 seconds
            cap.release()
            cv2.destroyAllWindows()

            print(f"Attempts left: {attempt_counter}")
            if attempt_counter == 0:
                show_too_many_attempts_dialog()
                exit()
    return False

def show_too_many_attempts_dialog():
    app = QtWidgets.QApplication([])
    QtWidgets.QMessageBox.critical(None, "Too Many Attempts", "You have exceeded the maximum number of attempts.")
    app.exec_()
    # Ensure the application exits properly
    QtCore.QCoreApplication.quit()

class HandGestureMouseKeyboardController:
    def __init__(self):
        # Initialize video capture
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.frame_width, self.frame_height = 1280, 720

        # Initialize MediaPipe Hands
        self.hands = mp.solutions.hands.Hands(max_num_hands=2, min_detection_confidence=0.8, min_tracking_confidence=0.8)
        self.mp_drawing = mp.solutions.drawing_utils

        # Mouse setup
        self.screen_width, self.screen_height = pyautogui.size()
        self.rect_width = int(self.frame_width * 0.7)
        self.rect_height = int(self.frame_height * 0.7)
        self.rect_x = (self.frame_width - self.rect_width) // 2
        self.rect_y = (self.frame_height - self.rect_height) // 2
        self.x_scale = self.screen_width / self.rect_width
        self.y_scale = self.screen_height / self.rect_height
        self.ema_alpha = 0.2  # Exponential Moving Average smoothing factor
        self.ema_x, self.ema_y = None, None  # EMA coordinates
        self.click_delay = 0.5  
        self.right_click_delay = 0.5  
        self.alt_tab_delay = 1.0 
        self.last_left_click_time = 0
        self.last_right_click_time = 0
        self.last_alt_tab_time = 0
        self.scrolling = False
        self.scroll_start_y = 0
        self.scroll_sensitivity = 50

        # Keyboard setup
        self.keys = [
            ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
            ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";"],
            ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "/"],
            ["CA", "SP", "BS", "<-", "->"]
        ]
        self.finalText = ""
        self.cursor_pos = 0
        self.caps_lock = False
        self.lastClickTime = 0
        self.buttonList = []
        for i, row in enumerate(self.keys):
            for j, key in enumerate(row):
                x = 100 * j + 50
                y = 100 * i + 50
                self.buttonList.append(Button([x, y], key))

        # Mode flags
        self.keyboard_mode = False
        self.mouse_mode = False

    def drawAll(self, img):
        """Draw the virtual keyboard buttons and text display."""
        for button in self.buttonList:
            x, y = button.pos
            w, h = button.size
            color = (255, 0, 255)
            if button.text == "CA":
                color = (255, 0, 0) if self.caps_lock else (255, 0, 255)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, cv2.FILLED)
            cv2.putText(img, button.text, (x + 20, y + 65), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
        display_text = self.finalText[:self.cursor_pos] + "|" + self.finalText[self.cursor_pos:]
        cv2.rectangle(img, (50, 500), (700, 600), (175, 0, 175), cv2.FILLED)
        cv2.putText(img, display_text, (60, 580), cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 5)
        return img

    def calculate_distance(self, point1, point2):
        """Calculate the Euclidean distance between two landmark points."""
        return math.sqrt((point2.x - point1.x) ** 2 + (point2.y - point1.y) ** 2)

    def fingers_up(self, landmarks, hand_label):
        """Determine which fingers are up based on landmarks and hand label."""
        finger_tips = [4, 8, 12, 16, 20]
        finger_bases = [3, 6, 10, 14, 18]
        fingers = []
        if hand_label == "Right":
            fingers.append(landmarks[finger_tips[0]].x < landmarks[finger_bases[0]].x)
        else:
            fingers.append(landmarks[finger_tips[0]].x > landmarks[finger_bases[0]].x)
        for tip, base in zip(finger_tips[1:], finger_bases[1:]):
            fingers.append(landmarks[tip].y < landmarks[base].y)
        return [1 if finger else 0 for finger in fingers]

    def move_mouse(self, target_x, target_y):
        """Move the mouse cursor using Exponential Moving Average for smoothing."""
        if self.ema_x is None or self.ema_y is None:
            self.ema_x, self.ema_y = target_x, target_y
        self.ema_x = self.ema_alpha * target_x + (1 - self.ema_alpha) * self.ema_x
        self.ema_y = self.ema_alpha * target_y + (1 - self.ema_alpha) * self.ema_y
        pyautogui.moveTo(int(self.ema_x), int(self.ema_y))

    def process_frame(self, frame):
        """Process each frame to detect hand gestures and control mouse/keyboard."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        # Reset mode flags
        self.keyboard_mode = False
        self.mouse_mode = False
        
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                landmarks = hand_landmarks.landmark
                label = handedness.classification[0].label

                self.mp_drawing.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
                
                # Assign mode based on hand label as per original code
                if label == "Right":
                    self.keyboard_mode = True
                elif label == "Left":
                    self.mouse_mode = True

                if label == "Right" and self.keyboard_mode:
                    index_tip = landmarks[8]
                    middle_tip = landmarks[12]
                    index_x = int(index_tip.x * self.frame_width)
                    index_y = int(index_tip.y * self.frame_height)
                    
                    for button in self.buttonList:
                        x, y = button.pos
                        w, h = button.size
                        # Dynamic UI: Change color and size on hover
                        if x < index_x < x + w and y < index_y < y + h:
                            cv2.rectangle(frame, (x-5, y-5), (x+w+5, y+h+5), (175,0,175), cv2.FILLED)
                            cv2.putText(frame, button.text, (x+20, y+65), cv2.FONT_HERSHEY_PLAIN, 4, (255,255,255), 4)

                            distance = self.calculate_distance(index_tip, middle_tip)
                            
                            if distance < 0.03:
                                current_time = time.time()
                                if current_time - self.lastClickTime > 0.5:
                                    self.lastClickTime = current_time
                                    if button.text == "CA":
                                        self.caps_lock = not self.caps_lock
                                    elif button.text == "SP":
                                        self.finalText = self.finalText[:self.cursor_pos] + " " + self.finalText[self.cursor_pos:]
                                        self.cursor_pos += 1
                                    elif button.text == "BS":
                                        if self.cursor_pos > 0:
                                            self.finalText = self.finalText[:self.cursor_pos - 1] + self.finalText[self.cursor_pos:]
                                            self.cursor_pos -= 1
                                    elif button.text == "<-":
                                        if self.cursor_pos > 0:
                                            self.cursor_pos -= 1
                                    elif button.text == "->":
                                        if self.cursor_pos < len(self.finalText):
                                            self.cursor_pos += 1
                                    else:
                                        char = button.text.upper() if self.caps_lock else button.text.lower()
                                        self.finalText = self.finalText[:self.cursor_pos] + char + self.finalText[self.cursor_pos:]
                                        self.cursor_pos += 1
                                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), cv2.FILLED)
                                    cv2.putText(frame, button.text, (x + 20, y + 65), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
                                    time.sleep(0.15)
                        else:
                            # Dynamic UI: Reset button color if not hovered
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,255), cv2.FILLED)
                            cv2.putText(frame, button.text, (x+20, y+65), cv2.FONT_HERSHEY_PLAIN, 4, (255,255,255), 4)
                        
                elif label == "Left" and self.mouse_mode:
                    fingers = self.fingers_up(landmarks, label)
                    
                    # On-Screen Indicators: Scroll gestures
                    # Scroll up when thumb is up and others are down
                    if fingers == [1, 0, 0, 0, 0]:
                        pyautogui.scroll(100)
                        cv2.putText(frame, "SCROLL UP", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    # Scroll down when all fingers are down
                    elif fingers == [0, 0, 0, 0, 0]:
                        pyautogui.scroll(-50)
                        cv2.putText(frame, "SCROLL DOWN", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    
                    # Mouse movement
                    if fingers == [0, 1, 0, 0, 0]:
                        index_tip = landmarks[8]
                        if (self.rect_x <= index_tip.x * self.frame_width <= self.rect_x + self.rect_width and
                            self.rect_y <= index_tip.y * self.frame_height <= self.rect_y + self.rect_height):
                            screen_x = int((index_tip.x * self.frame_width - self.rect_x) * self.x_scale)
                            screen_y = int((index_tip.y * self.frame_height - self.rect_y) * self.y_scale)
                            self.move_mouse(screen_x, screen_y)

                    # Right-click: Thumb and index up
                    if fingers[0] == 1 and fingers[1] == 1:
                        thumb_tip = landmarks[4]
                        index_tip = landmarks[8]
                        distance = self.calculate_distance(thumb_tip, index_tip)
                        if distance < 0.05:
                            current_time = time.time()
                            if current_time - self.last_right_click_time > self.right_click_delay:
                                print("Right click triggered")
                                pyautogui.click(button='right')
                                self.last_right_click_time = current_time

                    # Left-click to open files/folders: Index and middle up
                    if fingers[1] == 1 and fingers[2] == 1:
                        index_tip = landmarks[8]
                        middle_tip = landmarks[12]
                        distance = self.calculate_distance(index_tip, middle_tip)
                        
                        if distance < 0.03:
                            current_time = time.time()
                            if current_time - self.last_left_click_time > self.click_delay:
                                print("Left click triggered")
                                pyautogui.doubleClick()
                                self.last_left_click_time = current_time

                    # New gesture for Alt+Tab: Index and pinky up and close
                    if fingers[1] == 1 and fingers[4] == 1:
                        index_tip = landmarks[8]
                        pinky_tip = landmarks[20]
                        distance = self.calculate_distance(index_tip, pinky_tip)

                        if distance < 0.1:
                            current_time = time.time()
                            if current_time - self.last_alt_tab_time > self.alt_tab_delay:
                                print("Alt+Tab triggered")
                                pyautogui.hotkey('alt', 'tab')
                                # On-Screen Indicators: Alt+Tab
                                cv2.putText(frame, "SWITCHING WINDOWS", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                                self.last_alt_tab_time = current_time
        return frame

    def run(self):
        """Main loop to run the controller."""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            frame = self.process_frame(frame)

            if self.mouse_mode:
                cv2.rectangle(frame, (self.rect_x, self.rect_y), 
                              (self.rect_x + self.rect_width, self.rect_y + self.rect_height), (0, 255, 0), 2)
            
            if self.keyboard_mode:
                frame = self.drawAll(frame)

            mode_text = "Keyboard" if self.keyboard_mode else "Mouse" if self.mouse_mode else "None"
            cv2.putText(frame, f"Mode: {mode_text}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Virtual Mouse and Keyboard", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    if not face_recognition_auth():
        show_too_many_attempts_dialog()
    else:
        controller = HandGestureMouseKeyboardController()
        controller.run()