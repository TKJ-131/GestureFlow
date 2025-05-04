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

        # Convert landmarks to flat input format
        keypoints = []
        for id, x, y in lmList:
            keypoints.extend([x, y])
        keypoints = np.array(keypoints).reshape(1, -1)

        # Predict letter
        prediction = model.predict(keypoints)[0]

        return jsonify({'letter': prediction})

    except Exception as e:
        print("Error:", e)
        return jsonify({'letter': '?'})

if __name__ == '__main__':
    app.run(debug=True)
