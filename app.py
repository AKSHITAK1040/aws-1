import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

CASCADE_PATH = "haarcascade_frontalface_default.xml"

st.title(" Ethnicity Prediction")

# Load face detector
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
if face_cascade.empty():
    st.error("Failed to load Haar cascade. Ensure haarcascade_frontalface_default.xml is in your directory.")
    st.stop()

# Load YOLOv8 model once and cache it
@st.cache_resource
def load_model():
    return YOLO('yolo.pt')  # Replace with your model path

model = load_model()

# Camera input
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
        faces = np.array(faces)
        if faces.ndim == 1 and faces.shape[0] == 4:
            faces = faces.reshape(1, 4)

        # Select largest face
        (x, y, w, h) = sorted(faces, key=lambda b: b[2]*b[3], reverse=True)[0]
        face_img = img_cv[y:y+h, x:x+w]

        # Resize for faster inference
        face_img = cv2.resize(face_img, (224, 224))
        st.image(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB), caption="Focused Face", use_column_width=True)

        st.write("Predicting ethnicity...")

        # YOLO accepts NumPy array directly
        results = model(face_img, verbose=False)

        class_names = [
            'Black', 'East Asian', 'Indian', 'Latino_Hispanic',
            'Middle Eastern', 'Southeast Asian', 'White'
        ]
        top_class_idx = int(results[0].probs.top1)
        top_class = class_names[top_class_idx]
        st.success(f"Predicted Ethnicity: {top_class}")

        # Optional: show probabilities
        st.write("Class Probabilities:")
        probs_dict = {class_names[i]: float(results[0].probs.data[i]) for i in range(len(class_names))}
        st.json(probs_dict)
