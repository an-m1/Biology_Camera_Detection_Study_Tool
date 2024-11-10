import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Function to detect the blue pen tip precisely
def detect_blue_pen_tip(frame):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Define HSV range for blue color
    lower_blue = np.array([100, 150, 50])
    upper_blue = np.array([140, 255, 255])
    mask = cv2.inRange(hsv_frame, lower_blue, upper_blue)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Select the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        if 300 < cv2.contourArea(largest_contour) < 2000:  # Size constraints for pen tip
            # Find extreme points
            topmost = tuple(largest_contour[largest_contour[:, :, 1].argmin()][0])
            return topmost
    return None

# Define the list of keypoints to ignore
ignored_keypoints = [
    mp_pose.PoseLandmark.LEFT_EYE_INNER,
    mp_pose.PoseLandmark.LEFT_EYE_OUTER,
    mp_pose.PoseLandmark.RIGHT_EYE_OUTER,
    mp_pose.PoseLandmark.RIGHT_EYE_INNER,
    mp_pose.PoseLandmark.MOUTH_LEFT,
    mp_pose.PoseLandmark.MOUTH_RIGHT,
]

# Load images for specific keypoints
image_left_shoulder = cv2.imread('/Users/ankitmodhera/Desktop/camera_biology_study/images/left_shoulder.jpg')
image_right_shoulder = cv2.imread('/Users/ankitmodhera/Desktop/camera_biology_study/images/right_shoulder.png')
image_nose = cv2.imread('/Users/ankitmodhera/Desktop/camera_biology_study/images/nose.png')

# Dictionary to map keypoints to images
keypoint_images = {
    "LEFT_SHOULDER": image_left_shoulder,
    "RIGHT_SHOULDER": image_right_shoulder,
    "NOSE": image_nose,
}

# Track the currently displayed keypoint
current_body_part = None

# Start video capture with macOS-compatible backend
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
if not cap.isOpened():
    print("Camera not accessible! Check permissions.")
    exit()

print("Camera is working! Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame!")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    # Detect pose landmarks if a human is detected
    landmarks = None
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Detect the blue pen tip
    pen_tip = detect_blue_pen_tip(frame)

    # If landmarks and pen tip are detected, find the closest body part
    if pen_tip and landmarks:
        h, w, _ = frame.shape
        body_part = None
        min_distance = float('inf')

        # Iterate through all major body parts
        for idx, landmark in enumerate(landmarks):
            # Skip ignored keypoints
            if mp_pose.PoseLandmark(idx) in ignored_keypoints:
                continue

            if landmark.visibility > 0.5:  # Consider only visible landmarks
                x, y = int(landmark.x * w), int(landmark.y * h)
                distance = np.linalg.norm(np.array(pen_tip) - np.array([x, y]))
                
                # Only consider it "pointing" if the pen tip is close to the body part
                if distance < 50:  # Threshold for "pointing"
                    if distance < min_distance:
                        min_distance = distance
                        body_part = mp_pose.PoseLandmark(idx).name

        # Display the image as long as the pen is pointing at the body part
        if body_part and body_part in keypoint_images and keypoint_images[body_part] is not None:
            if current_body_part != body_part:
                current_body_part = body_part
                print(f"Blue pen is pointing at: {body_part}")
                cv2.imshow("Keypoint Detected", keypoint_images[body_part])

        # Close the image window if the pen is no longer pointing at a keypoint
        elif current_body_part:
            current_body_part = None
            cv2.destroyWindow("Keypoint Detected")

        # Highlight the closest body part and pen tip
        cv2.circle(frame, pen_tip, 5, (255, 0, 0), -1)  # Blue pen tip
        if body_part:
            cv2.putText(frame, f"Pointing at: {body_part}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the result
    cv2.imshow("Human Pose and Pen Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
