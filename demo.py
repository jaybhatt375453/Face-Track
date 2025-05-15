import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
from skimage.feature import local_binary_pattern
import time
import requests
from geopy.distance import geodesic

# Initialize the face detector and predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks (1).dat")

# Functions to calculate eye and mouth aspect ratios
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

# Function to get the current location using IP
def get_current_location():
    try:
        response = requests.get('https://ipinfo.io/json')
        data = response.json()
        
        if 'loc' in data:
            coords = data['loc'].split(',')
            current_lat = float(coords[0])
            current_long = float(coords[1])
            return current_lat, current_long
        else:
            print("Location data not available")
            return None, None
    except Exception as e:
        print(f"Error getting location: {e}")
        return None, None

# Function to verify the location
def verify_location(current_lat, current_long, expected_lat, expected_long, max_distance_km=0.5):
    if current_lat is None or current_long is None:
        return False, "Location unavailable"
    
    current_coords = (current_lat, current_long)
    expected_coords = (expected_lat, expected_long)
    
    # Calculate distance between current and expected coordinates
    distance = geodesic(current_coords, expected_coords).kilometers
    
    if distance <= max_distance_km:
        return True, f"Location verified (±{distance:.2f} km)"
    else:
        return False, f"Location mismatch ({distance:.2f} km away)"

# Expected university location coordinates
EXPECTED_LATITUDE = 24.08900
EXPECTED_LONGITUDE = 72.393500
LOCATION_TOLERANCE_KM = 0.5  # 500 meters tolerance

# Threshold values for liveness detection
EYE_AR_THRESHOLD = 0.25
MOUTH_AR_THRESHOLD = 0.6
EYE_BLINK_CONSEC_FRAMES = 2
MOUTH_MOVE_CONSEC_FRAMES = 2
MOVEMENT_THRESHOLD = 5
TIME_WINDOW = 10  # seconds

# Initialize counters and state variables
eye_blink_counter = 0
mouth_move_counter = 0
movement_count = 0
start_time = time.time()
location_verified = False
location_message = "Checking location..."

# Start video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not access webcam.")
    exit()

# Get location at startup
current_lat, current_long = get_current_location()
location_verified, location_message = verify_location(
    current_lat, current_long, 
    EXPECTED_LATITUDE, EXPECTED_LONGITUDE,
    LOCATION_TOLERANCE_KM
)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    current_time = time.time()
    elapsed_time = current_time - start_time

    # Display location information at the bottom of the frame
    cv2.putText(frame, f"Location: {location_message}", 
                (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, (0, 255, 0) if location_verified else (0, 0, 255), 2)
    
    # Display current coordinates if available
    if current_lat is not None and current_long is not None:
        cv2.putText(frame, f"Current: {current_lat:.6f}, {current_long:.6f}", 
                    (10, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Expected: {EXPECTED_LATITUDE:.6f}, {EXPECTED_LONGITUDE:.6f}", 
                    (10, frame.shape[0] - 80), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (255, 255, 255), 1)

    for face in faces:
        landmarks = predictor(gray, face)

        # Extract eye and mouth landmarks
        left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
        right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])
        mouth = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 68)])
        
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

        # Check for liveness after defined time window
        if elapsed_time > TIME_WINDOW:
            liveness_confirmed = movement_count >= MOVEMENT_THRESHOLD
            
            # Combined verification of liveness and location
            if liveness_confirmed and location_verified:
                verification_message = "Verification Complete: Real Person at Expected Location"
                color = (0, 255, 0)  # Green
            elif liveness_confirmed and not location_verified:
                verification_message = "Liveness Confirmed but Location Mismatch"
                color = (0, 165, 255)  # Orange
            elif not liveness_confirmed and location_verified:
                verification_message = "Location Verified but Liveness Failed"
                color = (0, 0, 255)  # Red
            else:
                verification_message = "Verification Failed: Not Live and Wrong Location"
                color = (0, 0, 255)  # Red
                
            cv2.putText(frame, verification_message, (face.left(), face.top()-30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            print(verification_message)
            
            # Reset counters and timer
            movement_count = 0
            start_time = current_time
            
            # Re-check location periodically
            if elapsed_time > 30:  # Check location every 30 seconds
                current_lat, current_long = get_current_location()
                location_verified, location_message = verify_location(
                    current_lat, current_long, 
                    EXPECTED_LATITUDE, EXPECTED_LONGITUDE,
                    LOCATION_TOLERANCE_KM
                )

        # Draw bounding box around the face
        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (255, 0, 0), 2)
        
        # Draw facial landmarks
        for (x, y) in np.vstack([left_eye, right_eye, mouth]):
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    # Display the frame
    cv2.imshow("Liveness and Location Verification", frame)
    
    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
