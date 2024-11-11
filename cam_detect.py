import cv2
import mediapipe as mp
import numpy as np
import os

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Function to detect the blue pen tip
def detect_blue_pen_tip(frame):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Define HSV range for blue color
    lower_blue = np.array([100, 150, 50])
    upper_blue = np.array([140, 255, 255])
    mask = cv2.inRange(hsv_frame, lower_blue, upper_blue)

    # Show the mask in the bottom-left corner
    small_mask = cv2.resize(mask, (200, 150))
    cv2.imshow("Blue Pen Mask", small_mask)
    cv2.moveWindow("Blue Pen Mask", 10, 600)  # Bottom-left

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        if 100 < cv2.contourArea(largest_contour) < 5000:  # Broader size constraints
            topmost = tuple(largest_contour[largest_contour[:, :, 1].argmin()][0])
            return topmost
    return None

# Define the list of keypoints to ignore
ignored_keypoints = [
    mp_pose.PoseLandmark.LEFT_EYE_INNER,
    mp_pose.PoseLandmark.LEFT_EYE_OUTER,
    mp_pose.PoseLandmark.RIGHT_EYE_OUTER,
    mp_pose.PoseLandmark.RIGHT_EYE_INNER,
]

# Load images for all keypoints
keypoint_image_paths = {
    "LEFT_SHOULDER": [
        "/Users/ankitmodhera/Desktop/cameraStudy_tool/images/left_shoulder.jpg",
        "/Users/ankitmodhera/Desktop/cameraStudy_tool/images/left_shoulder2.jpg"
    ],
    "RIGHT_SHOULDER": [
        "/Users/ankitmodhera/Desktop/cameraStudy_tool/images/right_shoulder.png"
    ],
    "NOSE": [
        "/Users/ankitmodhera/Desktop/cameraStudy_tool/images/nose.png",
        "/Users/ankitmodhera/Desktop/cameraStudy_tool/images/nose2.jpg"
    ],
    "MOUTH": [
        "/Users/ankitmodhera/Desktop/cameraStudy_tool/images/mouth.png",
        "/Users/ankitmodhera/Desktop/cameraStudy_tool/images/mouth1.png"
    ],
    "LEFT_EYE": [
        "/Users/ankitmodhera/Desktop/cameraStudy_tool/images/left_eye.png"
    ],
    "RIGHT_EYE": [
        "/Users/ankitmodhera/Desktop/cameraStudy_tool/images/right_eye.png"
    ],
    "LEFT_EAR": [
        "/Users/ankitmodhera/Desktop/cameraStudy_tool/images/left_ear.png"
    ],
    "RIGHT_EAR": [
        "/Users/ankitmodhera/Desktop/cameraStudy_tool/images/right_ear.jpg"
    ],
}

# Load and validate images
keypoint_images = {}
for keypoint, paths in keypoint_image_paths.items():
    keypoint_images[keypoint] = []
    for path in paths:
        if os.path.exists(path):
            image = cv2.imread(path)
            if image is not None:
                keypoint_images[keypoint].append(image)
        else:
            print(f"Error: Could not load image at {path}")

current_body_part = None

# Detection radius and setup
DETECTION_RADIUS = 100

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

    landmarks = None
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    pen_tip = detect_blue_pen_tip(frame)

    if pen_tip and landmarks:
        h, w, _ = frame.shape
        body_part = None
        min_distance = float('inf')

        # Calculate midpoint for the mouth
        mouth_midpoint = None
        if (
            landmarks[mp_pose.PoseLandmark.MOUTH_LEFT.value].visibility > 0.5
            and landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT.value].visibility > 0.5
        ):
            left_mouth = landmarks[mp_pose.PoseLandmark.MOUTH_LEFT.value]
            right_mouth = landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT.value]
            mouth_midpoint = (
                int((left_mouth.x + right_mouth.x) / 2 * w),
                int((left_mouth.y + right_mouth.y) / 2 * h),
            )

        # Ignore the hand holding the pen
        pen_hand_detected = None
        for idx, landmark in enumerate(landmarks):
            if idx in [mp_pose.PoseLandmark.LEFT_WRIST.value, mp_pose.PoseLandmark.RIGHT_WRIST.value]:
                x, y = int(landmark.x * w), int(landmark.y * h)
                distance = np.linalg.norm(np.array(pen_tip) - np.array([x, y]))
                if distance < DETECTION_RADIUS:
                    pen_hand_detected = mp_pose.PoseLandmark(idx).name
                    break

        # Iterate through all major body parts
        for idx, landmark in enumerate(landmarks):
            # Skip ignored keypoints
            if mp_pose.PoseLandmark(idx) in ignored_keypoints:
                continue

            if landmark.visibility > 0.5:
                x, y = int(landmark.x * w), int(landmark.y * h)
                distance = np.linalg.norm(np.array(pen_tip) - np.array([x, y]))
                if distance < DETECTION_RADIUS and distance < min_distance:
                    if mp_pose.PoseLandmark(idx).name != pen_hand_detected:  # Ignore pen hand
                        min_distance = distance
                        body_part = mp_pose.PoseLandmark(idx).name

        # Check the distance to the mouth midpoint
        if mouth_midpoint:
            mouth_distance = np.linalg.norm(np.array(pen_tip) - np.array(mouth_midpoint))
            if mouth_distance < DETECTION_RADIUS and mouth_distance < min_distance:
                min_distance = mouth_distance
                body_part = "MOUTH"

        # Log and display keypoints
        if body_part:
            print(f"Blue pen is pointing at: {body_part}")

            # Close all previously opened keypoint windows
            for i in range(len(keypoint_images.get(current_body_part, []))):
                cv2.destroyWindow(f"Keypoint Detected {i + 1}")

            current_body_part = body_part

            for i, img in enumerate(keypoint_images.get(body_part, [])):
                resized_img = cv2.resize(img, (400, 300))
                window_name = f"Keypoint Detected {i + 1}"
                if i == 0:
                    cv2.imshow(window_name, resized_img)
                    cv2.moveWindow(window_name, 1000, 10)  # Top-right
                elif i == 1:
                    cv2.imshow(window_name, resized_img)
                    cv2.moveWindow(window_name, 1000, 380)  # Bottom-right

        elif current_body_part:
            current_body_part = None
            cv2.destroyAllWindows()

        # Highlight pen tip and keypoints
        cv2.circle(frame, pen_tip, 10, (255, 0, 0), -1)
        if mouth_midpoint:
            cv2.circle(frame, mouth_midpoint, 10, (0, 255, 255), -1)  # Highlight mouth midpoint
        if body_part:
            cv2.putText(frame, f"Pointing at: {body_part}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Camera feed in top-left corner
    cv2.imshow("Human Pose and Pen Detection", frame)
    cv2.moveWindow("Human Pose and Pen Detection", 10, 10)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
