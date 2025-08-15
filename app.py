import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import json

# Load the pre-trained model and class mapping
model = tf.keras.models.load_model('ethnicity_classification_mobilenetv2.h5')
with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)
reverse_class_indices = {v: k for k, v in class_indices.items()}

# Load OpenCV Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def preprocess_face(frame, img_size=(128, 128)):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) > 0:
        x, y, w, h = faces[0]
        face = frame[y:y+h, x:x+w]
        img = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    else:
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, img_size)
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)

st.set_page_config(page_title="Ethnicity Predictor", layout="wide")
st.title("Ethnicity Predictor (Browser Webcam)")

st.write("Use your browser webcam to take a picture and predict ethnicity.")

uploaded_image = st.camera_input("Capture image")

if uploaded_image is not None:
    # Convert to OpenCV format
    image = np.array(Image.open(uploaded_image))
    
    # Predict ethnicity
    input_data = preprocess_face(image)
    pred = model.predict(input_data)
    pred_class = np.argmax(pred, axis=1)[0]
    pred_label = reverse_class_indices[pred_class]

    # Draw rectangle if face detected
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) > 0:
        x, y, w, h = faces[0]
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(image, f'Ethnicity: {pred_label}', (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(image, f'Ethnicity: {pred_label}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    # Display final image
    st.image(image, channels="BGR")
