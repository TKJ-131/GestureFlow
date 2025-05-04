import cv2
import csv
import os
from HandTrackingModule import HandDetector
import string

# Setup
detector = HandDetector(detectionCon=0.7)
cap = cv2.VideoCapture(0)
output_file = 'sign_data.csv'

# Check if file exists; if not, create with header
if not os.path.exists(output_file):
    with open(output_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        header = ['label']
        for i in range(21):  # 21 hand landmarks
            header += [f'x{i}', f'y{i}']
        writer.writerow(header)

print("Press A-Z to label signs. Press ESC to exit.")

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if lmList:
        cv2.putText(img, "Press A-Z to save this hand sign", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

    cv2.imshow("Collect ASL Data", img)
    key = cv2.waitKey(1)

    if key == 27:  # ESC
        break
    elif 65 <= key <= 90:  # A–Z keys
        label = chr(key)
        if lmList:
            x_coords = [x for _, x, y in lmList]
            y_coords = [y for _, x, y in lmList]
            row = [label] + [coord for pair in zip(x_coords, y_coords) for coord in pair]
            with open(output_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
            print(f"Saved: {label}")

cap.release()
cv2.destroyAllWindows()
