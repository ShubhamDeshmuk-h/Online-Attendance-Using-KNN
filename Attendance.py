import cv2
import numpy as np
import os
import csv
import time
import pickle
from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime

# Start webcam
video = cv2.VideoCapture(0)

# Load face detector
facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Load label and face data
with open('data/names.pkl', 'rb') as w:
    LABELS = pickle.load(w)

with open('data/face_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

# Train model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

# Load background image
imgbackground = cv2.imread('bg.png')

COL_NAMES = ['NAME', 'TIME']

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        crop_img = frame[y:y + h, x:x + w]
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
        output = knn.predict(resized_img)

        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
        exist = os.path.isfile("Attendance/Attendance_" + date + '.csv')

        # Drawing rectangle and text
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)
        cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)
        cv2.putText(frame, str(output[0]), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        attendance = [str(output[0]), str(timestamp)]

        # Overlay on background image
        if imgbackground is not None and imgbackground.shape[0] >= 642 and imgbackground.shape[1] >= 690:
            imgbackground[162:162 + 480, 50:50 + 640] = frame
            cv2.imshow("frame", imgbackground)
        else:
            cv2.imshow("frame", frame)  # fallback if background missing or too small

        k = cv2.waitKey(1)
        if k == ord('o'):
            time.sleep(1)
            os.makedirs("Attendance", exist_ok=True)
            file_path = f'Attendance/Attendance_{date}.csv'
            with open(file_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                if not exist:
                    writer.writerow(COL_NAMES)
                writer.writerow(attendance)

        if k == ord('q'):
            break

video.release()
cv2.destroyAllWindows()
