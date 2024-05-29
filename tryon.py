from flask import Flask, render_template, request
import threading
import cv2
import numpy as np
import time
from cvzone.PoseModule import PoseDetector

app = Flask(__name__)
camera_running = False
camera_thread = None

# Known average shoulder width in cm
average_shoulder_width_cm = 38
scaling_factor_adjustment = 0.9  # Adjust this factor to calibrate height estimation

# Variables for averaging height over time
heights = []
start_time = time.time()
interval = 2  # Interval in seconds
last_displayed_height = None  # Store the last displayed height

# Button properties
button_position = (10, 10)
button_size = (50, 30)
button_color = (0, 0, 255)
button_text = "Exit"
button_text_color = (255, 255, 255)

def draw_button(img, position, size, color, text, text_color):
    x, y = position
    w, h = size
    cv2.rectangle(img, (x, y), (x + w, y + h), color, -1)
    cv2.putText(img, text, (x + 5, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

def is_inside_button(x, y, position, size):
    px, py = position
    w, h = size
    return px <= x <= px + w and py <= y <= py + h

def estimate_height(lmList):
    if len(lmList) > 24:  # Ensure that all necessary points are detected
        shoulder_left = lmList[11][1:3]
        shoulder_right = lmList[12][1:3]
        hip_left = lmList[23][1:3]
        hip_right = lmList[24][1:3]
        ankle_left = lmList[27][1:3]
        ankle_right = lmList[28][1:3]
        
        # Calculate pixel distances
        shoulder_width_px = np.linalg.norm(np.array(shoulder_left) - np.array(shoulder_right))
        shoulder_to_ankle_left_px = np.linalg.norm(np.array(shoulder_left) - np.array(ankle_left))
        shoulder_to_ankle_right_px = np.linalg.norm(np.array(shoulder_right) - np.array(ankle_right))
        
        # Average the shoulder-to-ankle distance for left and right sides
        avg_shoulder_to_ankle_px = (shoulder_to_ankle_left_px + shoulder_to_ankle_right_px) / 2
        
        # Calculate the scaling factor
        if shoulder_width_px > 0:
            scaling_factor = (average_shoulder_width_cm / shoulder_width_px) * scaling_factor_adjustment
            
            # Estimate height in cm
            estimated_height_cm = avg_shoulder_to_ankle_px * scaling_factor
            
            return estimated_height_cm
    return None

def start_camera():
    global camera_running, heights, start_time, last_displayed_height
    cap = cv2.VideoCapture(0)  # Capture from camera
    detector = PoseDetector()

    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Image", 500, 500)
    
    while camera_running:
        success, img = cap.read()
        if not success:
            break

        # Mirror the image horizontally
        img = cv2.flip(img, 1)

        img = detector.findPose(img)
        lmList, bboxInfo = detector.findPosition(img, bboxWithHands=False, draw=False)
        
        if lmList:
            estimated_height = estimate_height(lmList)
            if estimated_height:
                heights.append(estimated_height)
                
                # Update every interval seconds
                if time.time() - start_time > interval:
                    average_height = np.mean(heights)
                    heights = []
                    start_time = time.time()
                    last_displayed_height = average_height
        
        if last_displayed_height:
            cv2.putText(img, f"Estimated Height: {last_displayed_height:.2f} cm", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Draw exit button
        draw_button(img, button_position, button_size, button_color, button_text, button_text_color)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start():
    global camera_running, camera_thread
    if not camera_running:
        camera_running = True
        camera_thread = threading.Thread(target=start_camera)
        camera_thread.start()
    return 'Camera started'

@app.route('/stop', methods=['POST'])
def stop():
    global camera_running, camera_thread
    camera_running = False
    if camera_thread is not None:
        camera_thread.join()
    return 'Camera stopped'

if __name__ == '__main__':
    app.run(debug=True)
