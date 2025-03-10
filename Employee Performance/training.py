import cv2
import numpy as np
import os

# Initialize OpenCV's face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load OpenCV's Haar Cascade (for face detection)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Path to employees folder
employee_folder = r"E:\Nandhika\Billiance\Employee performance\Employee photos"

employee_images = []
employee_labels = []
employee_names = {}

id_counter = 0

if not os.path.exists(employee_folder):
    print(f"Error: Folder '{employee_folder}' does not exist.")
    exit()

# Process each image
for file in os.listdir(employee_folder):
    if file.lower().endswith((".jpg", ".jpeg", ".png")):
        img_path = os.path.join(employee_folder, file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"Skipping {file}: Unable to load image.")
            continue

        # Resize large images
        if img.shape[0] > 1000 or img.shape[1] > 1000:
            img = cv2.resize(img, (500, 500))

        # Detect faces
        faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            print(f"Skipping {file}: No face detected.")
            continue

        for (x, y, w, h) in faces:
            face = img[y:y+h, x:x+w]  # Crop face
            employee_images.append(face)
            employee_labels.append(id_counter)

        employee_names[id_counter] = os.path.splitext(file)[0]  # Store name
        id_counter += 1

# Ensure training data is not empty
if len(employee_images) == 0:
    print("Error: No valid faces detected for training.")
    exit()

# Train the recognizer
recognizer.train(employee_images, np.array(employee_labels))

# Save the trained model
recognizer.save("trained_faces.yml")

print("Training complete. Employees loaded:", employee_names)

