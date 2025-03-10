import cv2
import numpy as np

# Load trained recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(r"E:\Nandhika\Billiance\Employee performance\trained_faces.yml")  # Load trained faces

# Load Haar Cascade (for face detection)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Employee names (same as in training)
employee_names = {
    0: "Employee1",
    1: "Employee2"
}

# Open webcam
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]  # Crop face
        
        # Recognize the face
        label, confidence = recognizer.predict(face)

        if confidence < 95:  # Lower confidence = better match
            name = employee_names.get(label, "Employee")  # Recognized employee
            color = (0, 255, 0)  # Green for known faces
        else:
            name = "Customer"  # Unrecognized face
            color = (255, 0, 0)  # Blue for unknown faces

        # Draw rectangle and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Face Recognition", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()