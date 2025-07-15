import cv2 as cv
import face_recognition
import numpy as np
import csv
import time
import os

# Initialize camera
video = cv.VideoCapture(0)

# Load student data from CSV
students_data = {}
students_list = []

print("Loading student data...")
with open('1.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        student_id = row[0]
        name = row[1].lower()
        phone = row[2]
        students_data[name] = phone
        students_list.append(name)

print(f"Loaded {len(students_list)} students: {students_list}")

# Load known faces
known_face_encodings = []
known_face_names = []

def load_face_from_image(image_path, student_name):
    try:
        image = face_recognition.load_image_file(image_path)
        face_encodings = face_recognition.face_encodings(image)
        if len(face_encodings) > 0:
            known_face_encodings.append(face_encodings[0])
            known_face_names.append(student_name)
            print(f"✓ Loaded face for {student_name}")
            return True
        else:
            print(f"✗ No face found in {image_path}")
            return False
    except Exception as e:
        print(f"✗ Error loading {image_path}: {e}")
        return False

faces_dir = "faces"
if not os.path.exists(faces_dir):
    os.makedirs(faces_dir)
    print(f"Created {faces_dir} directory")

print("\nLoading face images...")
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

for student_name in students_list:
    loaded = False
    for ext in image_extensions:
        image_path = os.path.join(faces_dir, f"{student_name}{ext}")
        if os.path.exists(image_path):
            if load_face_from_image(image_path, student_name):
                loaded = True
                break
    if not loaded:
        print(f"⚠ No image found for {student_name}")

print(f"\nTotal faces loaded: {len(known_face_encodings)}")
if len(known_face_encodings) == 0:
    print("ERROR: No faces loaded! Check your 'faces' folder.")
    exit()

# Attendance tracking
present_students = []
absent_students = students_list.copy()

print("\nStarting attendance system...")
print("Press ESC to exit and show attendance report")

while True:
    check, frame = video.read()
    if not check:
        print("Error: Cannot read from camera")
        break

    small_frame = cv.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv.cvtColor(small_frame, cv.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []  

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.45)  

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            if name in absent_students:
                present_students.append(name)
                absent_students.remove(name)
                print(f"✓ Attendance marked for {name.upper()}")

        face_names.append(name)  
    

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv.putText(frame, name.upper(), (left + 6, bottom - 6),
                   cv.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

    # Display counters
    cv.putText(frame, f"Present: {len(present_students)}", (10, 30),
               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv.putText(frame, f"Absent: {len(absent_students)}", (10, 70),
               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv.imshow("Face Attendance System", frame)

    key = cv.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break

    time.sleep(0.1)

# Attendance report
print("\n" + "="*50)
print("ATTENDANCE REPORT")
print("="*50)

print(f"\nPRESENT STUDENTS ({len(present_students)}):")
for student in present_students:
    phone = students_data.get(student, "N/A")
    print(f"✓ {student.upper()} - {phone}")

print(f"\nABSENT STUDENTS ({len(absent_students)}):")
for student in absent_students:
    phone = students_data.get(student, "N/A")
    print(f"✗ {student.upper()} - {phone}")

# Save report
timestamp = time.strftime("%Y%m%d_%H%M%S")
reports_folder = r"C:\Users\91933\OneDrive\Desktop\attendance system\attendance_reports"
if not os.path.exists(reports_folder):
    os.makedirs(reports_folder)
    print(f"Created directory: {reports_folder}")

attendance_file = os.path.join(reports_folder, f"attendance_{timestamp}.csv")

with open(attendance_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Name', 'Phone', 'Status', 'Date', 'Time'])
    current_date = time.strftime("%Y-%m-%d")
    current_time = time.strftime("%H:%M:%S")

    for student in present_students:
        phone = students_data.get(student, "N/A")
        writer.writerow([student.upper(), phone, 'Present', current_date, current_time])
    for student in absent_students:
        phone = students_data.get(student, "N/A")
        writer.writerow([student.upper(), phone, 'Absent', current_date, current_time])

print(f"\nAttendance saved to: {attendance_file}")

# Cleanup
video.release()
cv.destroyAllWindows()
print("\nAttendance system closed.")
