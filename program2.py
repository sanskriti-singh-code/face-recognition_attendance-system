import face_recognition
import numpy as np
import cv2 as cv
import csv
import os
from datetime import datetime

video_capture = cv.VideoCapture(0)

names_1 = ['tony', 'rahul', 'kareena', 'karishma', 'tamanna']
known_face_encodings = []
names = names_1  # Store names for reference

for name in names_1:
    img = face_recognition.load_image_file(f"project/photos1/{name}.jpeg")
    img_encoding = face_recognition.face_encodings(img)[0]
    known_face_encodings.append(img_encoding)

students = names.copy()

face_names = []

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

name_list = []
if os.path.exists(current_date + '.csv'):
    with open(current_date + '.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            name_list.append(row[0])


while True:
    _, frame = video_capture.read()
    small_frame = cv.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    
    face_names = []
    
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"  # Default name for unrecognized faces

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = names[best_match_index]
            if name in students and name not in name_list:  # Check if the name is not already in the list
                students.remove(name)
                current_time = now.strftime("%H-%M-%S")
                name_list.append(name)  # Add the name to the list in memory
                # Write the name and current_time to the file
                with open(current_date + '.csv', 'a', newline='') as f:
                    lnwriter = csv.writer(f)
                    lnwriter.writerow([name, current_time])
                

        face_names.append(name)

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2

        cv.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        font = cv.FONT_HERSHEY_DUPLEX
        cv.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    cv.imshow("attendance system", frame)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

print(face_names)
video_capture.release()
cv.destroyAllWindows()
f.close()
