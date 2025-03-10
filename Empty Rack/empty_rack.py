import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageDraw
from roboflow import Roboflow

# Initialize the Roboflow client
rf = Roboflow(api_key="5VOF2OcYifmOkClliopV")  # Replace with your actual API key
project = rf.workspace().project("empty-spaces-detection-in-shelf-data")
model = project.version(2).model

def upload_image():
    # Open a file dialog to select an image
    file_path = filedialog.askopenfilename()
    if not file_path:
        return

    # Load the image using PIL
    image = Image.open(file_path)
    draw = ImageDraw.Draw(image)

    # Perform inference using the Roboflow API
    result = model.predict(file_path, confidence=40, overlap=30).json()

    # Draw bounding boxes on the image
    for prediction in result['predictions']:
        x = prediction['x']
        y = prediction['y']
        width = prediction['width']
        height = prediction['height']
        confidence = prediction['confidence']
        class_name = prediction['class']

        # Calculate the bounding box coordinates
        left = x - width / 2
        top = y - height / 2
        right = x + width / 2
        bottom = y + height / 2

        # Draw the bounding box
        draw.rectangle([left, top, right, bottom], outline="red", width=2)
        draw.text((left, top - 10), f"{class_name} {confidence:.2f}", fill="red")

    # Display the image with bounding boxes
    image.thumbnail((800, 800))  # Resize to fit in the UI
    photo = ImageTk.PhotoImage(image)
    image_label.config(image=photo)
    image_label.image = photo  # Keep a reference to avoid garbage collection

# Create the main window
root = tk.Tk()
root.title("Empty Spaces Detection in Shelf Data")

upload_button = tk.Button(root, text="Upload Image", command=upload_image)
upload_button.pack(pady=20)

image_label = tk.Label(root)
image_label.pack()

root.mainloop()
