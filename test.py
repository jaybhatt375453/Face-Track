import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
from skimage.feature import local_binary_pattern
from skimage import color
import time

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks (1).dat")

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[2], mouth[10])
    B = dist.euclidean(mouth[4], mouth[8])
    C = dist.euclidean(mouth[0], mouth[6])
    return (A + B) / (2.0 * C)

# Threshold values
EYE_AR_THRESHOLD = 0.25
MOUTH_AR_THRESHOLD = 0.6
EYE_BLINK_CONSEC_FRAMES = 2
MOUTH_MOVE_CONSEC_FRAMES = 2
MOVEMENT_THRESHOLD = 5
TIME_WINDOW = 10  # seconds

# Initialize counters
eye_blink_counter = 0
mouth_move_counter = 0
movement_count = 0
start_time = time.time()

# Start video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not access webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    current_time = time.time()
    elapsed_time = current_time - start_time

    for face in faces:
        landmarks = predictor(gray, face)

        # Extract eye and mouth landmarks
        left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
        right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])
        mouth = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 68)])
        nose_tip = (landmarks.part(30).x, landmarks.part(30).y)  # Nose tip landmark

        # Compute aspect ratios
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0
        mar = mouth_aspect_ratio(mouth)

        # Eye blink detection
        if ear < EYE_AR_THRESHOLD:
            eye_blink_counter += 1
        else:
            if eye_blink_counter >= EYE_BLINK_CONSEC_FRAMES:
                movement_count += 1
                eye_blink_counter = 0

        # Mouth movement detection
        if mar > MOUTH_AR_THRESHOLD:
            mouth_move_counter += 1
        else:
            if mouth_move_counter >= MOUTH_MOVE_CONSEC_FRAMES:
                movement_count += 1
                mouth_move_counter = 0

        # Check for liveness after 10 seconds
        if elapsed_time > TIME_WINDOW:
            if movement_count >= MOVEMENT_THRESHOLD:
                cv2.putText(frame, "Liveness Confirmed! (Real)", (face.left(), face.top()-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                print("Liveness Confirmed! (Real)")
            else:
                cv2.putText(frame, "No Liveness Detected! (Fake)", (face.left(), face.top()-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                print("No Liveness Detected! (Fake)")
            # Reset counters and timer
            movement_count = 0
            start_time = current_time

        # Draw bounding box
        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (255, 0, 0), 2)
        
        # Draw facial landmarks
        for (x, y) in np.vstack([left_eye, right_eye, mouth]):
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    # Display the frame
    cv2.imshow("Liveness Detection", frame)
    
    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
