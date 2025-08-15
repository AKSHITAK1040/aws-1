import streamlit as st
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO

# Set the path to your Haar cascade file (ensure the .xml is in your working directory)
CASCADE_PATH = "haarcascade_frontalface_default.xml"

st.title(" Ethnicity Prediction")

# Load the face detector safely
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
if face_cascade.empty():
    st.error("Failed to load Haar cascade. Ensure haarcascade_frontalface_default.xml is in your directory.")
    st.stop()

# Load your YOLOv8 ethnicity model
model = YOLO('Race-CLS-FairFace_yolo11l.pt')  # Rename for your specific file

# Use Streamlit camera input for live photo capture
capture = st.camera_input("Take a photo")

if capture is not None:
    image = Image.open(capture)
    st.image(image, caption="Captured Image", use_column_width=True)

    st.write("Focusing on the largest face...")
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    faces = face_cascade.detectMultiScale(img_cv, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        st.error("No face detected in the photo. Please try again.")
    else:
        # Ensure output is always a list of rectangles
        faces = np.array(faces)
        if faces.ndim == 1 and faces.shape[0] == 4:
            faces = faces.reshape(1, 4)

        # Select the largest face
        (x, y, w, h) = sorted(faces, key=lambda b: b[2]*b[3], reverse=True)[0]
        face_img = img_cv[y:y+h, x:x+w]

        # Convert cropped face to PIL for YOLO
        face_img_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
        st.image(face_img_pil, caption="Focused Face", use_column_width=True)

        # Ethnicity classification
        st.write("Predicting ethnicity...")
        results = model(face_img_pil)
        st.write(results[0].probs.data)  # Probabilities per class

        class_names = [
            'Black', 'East Asian', 'Indian', 'Latino_Hispanic',
            'Middle Eastern', 'Southeast Asian', 'White'
        ]
        top_class_idx = int(results[0].probs.top1)
        top_class = class_names[top_class_idx]
        st.success(f"Predicted Ethnicity: {top_class}")
