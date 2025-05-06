import cv2
import csv
import os
from HandTrackingModule import HandDetector

SAMPLES_PER_CLASS = 250
detector = HandDetector(detectionCon=0.7)
cap = cv2.VideoCapture(0)
output_file = 'sign_data.csv'

def normalize_landmarks(lmList):
    if not lmList:
        return []
    x0, y0 = lmList[0][1], lmList[0][2]
    coords = [(x - x0, y - y0) for (_, x, y) in lmList]
    flat = [item for tup in coords for item in tup]
    max_abs = max([abs(val) for val in flat]) or 1
    normalized = [(x / max_abs, y / max_abs) for (x, y) in coords]
    return [coord for tup in normalized for coord in tup]

if not os.path.exists(output_file):
    with open(output_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        header = ['label']
        for i in range(21): # 21 hand landmarks
            header += [f'x{i}', f'y{i}']
        writer.writerow(header)

collecting = False
current_label = ''
collected = 0

print("Press a letter key (A-Z) to start collecting samples for that sign.")
print("Press ESC to exit.")

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if collecting:
        cv2.putText(img, f"Collecting: '{current_label}' [{collected}/{SAMPLES_PER_CLASS}]",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        if lmList:
            row = [current_label]
            norm_coords = normalize_landmarks(lmList)
            row.extend(norm_coords)
            with open(output_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
            collected += 1
            if collected >= SAMPLES_PER_CLASS:
                collecting = False
                print(f"Collected {SAMPLES_PER_CLASS} samples for '{current_label}'.")
    else:
        cv2.putText(img, "Press A-Z to start collection | ESC to exit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

    cv2.imshow("Collect ASL Data", img)
    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break
    elif 65 <= key <= 90 or 97 <= key <= 122:  # A-Z or a-z
        current_label = chr(key).upper()
        collecting = True
        collected = 0
        print(f"Started collecting for '{current_label}'.")

cap.release()
cv2.destroyAllWindows()
print("Data collection finished.")