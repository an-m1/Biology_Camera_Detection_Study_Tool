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
            leftmost = tuple(largest_contour[largest_contour[:, :, 0].argmin()][0])
            rightmost = tuple(largest_contour[largest_contour[:, :, 0].argmax()][0])
            topmost = tuple(largest_contour[largest_contour[:, :, 1].argmin()][0])
            bottommost = tuple(largest_contour[largest_contour[:, :, 1].argmax()][0])
            
            # Assume the ballpoint side is the topmost point
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

        # Check the distance to the mouth midpoint
        if mouth_midpoint:
            mouth_distance = np.linalg.norm(np.array(pen_tip) - np.array(mouth_midpoint))
            if mouth_distance < 50 and mouth_distance < min_distance:
                min_distance = mouth_distance
                body_part = "MOUTH"

        # Log the detected body part
        if body_part:
            print(f"Blue pen is pointing at: {body_part}")

        # Highlight the closest body part and pen tip
        cv2.circle(frame, pen_tip, 5, (255, 0, 0), -1)  # Blue pen tip
        if mouth_midpoint:
            cv2.circle(frame, mouth_midpoint, 5, (0, 255, 255), -1)  # Highlight mouth midpoint
        if body_part:
            cv2.putText(frame, f"Pointing at: {body_part}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the result
    cv2.imshow("Human Pose and Pen Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
