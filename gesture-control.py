from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
import pickle
from HandTrackingModule import HandDetector

# Create Flask app
app = Flask(__name__)

# Load the trained sign language classifier
model = pickle.load(open('model/sign_classifier.pkl', 'rb'))

# Initialize hand detector
detector = HandDetector(detectionCon=0.7)

@app.route('/')
def index():
    return render_template('index.html')

def normalize_landmarks(lmList):
    if not lmList:
        return []
    x0, y0 = lmList[0][1], lmList[0][2]
    coords = [(x - x0, y - y0) for (_, x, y) in lmList]
    flat = [item for tup in coords for item in tup]
    max_abs = max([abs(val) for val in flat]) or 1
    normalized = [(x / max_abs, y / max_abs) for (x, y) in coords]
    return [coord for tup in normalized for coord in tup]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Receive base64 image from frontend
        data = request.json['image']
        encoded_data = data.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Detect hand
        img = detector.findHands(img)
        lmList = detector.findPosition(img, draw=False)

        # If no hand detected
        if not lmList:
            return jsonify({'letter': '?'})
        norm_coords = normalize_landmarks(lmList)
        keypoints = np.array(norm_coords).reshape(1, -1)
        prediction = model.predict(keypoints)[0]
        return jsonify({'letter': prediction})
    except Exception as e:
        print("Error:", e)
        return jsonify({'letter': '?'})

if __name__ == '__main__':
    app.run(debug=True)