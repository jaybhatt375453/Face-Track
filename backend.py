import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
import base64
import io
from PIL import Image
from flask import Flask, jsonify, request, render_template

# Initialize Flask app
app = Flask(__name__)

# Load face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor_path = "shape_predictor_68_face_landmarks (1).dat"
predictor = dlib.shape_predictor(predictor_path)

# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Function to calculate Mouth Aspect Ratio (MAR)
def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[2], mouth[10])
    B = dist.euclidean(mouth[4], mouth[8])
    C = dist.euclidean(mouth[0], mouth[6])
    return (A + B) / (2.0 * C)

# Route for serving the main HTML page
@app.route('/')
def index():
    return render_template('index.html')

# Route for checking liveness from frames
@app.route('/check-liveness', methods=['POST'])
def check_liveness():
    try:
        data = request.json
        frames = data.get('frames', [])

        if len(frames) < 5:
            return jsonify({
                'success': False,
                'error': 'Not enough frames provided (minimum 5)'
            })

        ear_values = []
        mar_values = []
        blink_counter = 0
        last_ear = None
        movement_detected = False

        for frame_data in frames:
            image_binary = base64.b64decode(frame_data.split(',')[1])
            pil_image = Image.open(io.BytesIO(image_binary))
            frame = np.array(pil_image)

            if frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)

            if not faces:
                continue

            face = faces[0]
            landmarks = predictor(gray, face)

            left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
            right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])
            mouth = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 68)])

            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0
            mar = mouth_aspect_ratio(mouth)

            ear_values.append(ear)
            mar_values.append(mar)

            if last_ear is not None:
                if last_ear > 0.15 and ear <= 0.15:
                    blink_counter += 1

            if ear < 0.25 or mar > 0.6:
                movement_detected = True

            last_ear = ear

        ear_variance = np.var(ear_values) if ear_values else 0
        mar_variance = np.var(mar_values) if mar_values else 0

        is_live = (blink_counter >= 5 and movement_detected and ear_variance > 0.001 and mar_variance > 0.001)

        confidence = 0.5
        if blink_counter >= 5:
            confidence += 0.2
        if movement_detected:
            confidence += 0.1
        if ear_variance > 0.001:
            confidence += 0.1
        if mar_variance > 0.001:
            confidence += 0.1

        return jsonify({
            'success': True,
            'is_live': is_live,
            'confidence': min(confidence, 1.0),
            'blink_count': blink_counter,
            'movement_detected': movement_detected,
            'ear_variance': float(ear_variance),
            'mar_variance': float(mar_variance)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

# Main function to run Flask server
if __name__ == '__main__':
    app.run(debug=True)
