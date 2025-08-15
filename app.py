import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import json

# Load model and class mapping
model = tf.keras.models.load_model('ethnicity_classification_mobilenetv2.h5')
with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)
reverse_class_indices = {v: k for k, v in class_indices.items()}

# Load OpenCV Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def preprocess_face(frame, img_size=(128,128)):
    # Detect face; if found, crop to the face, else use the full frame
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

st.title('Live Webcam Ethnicity Predictor with Face Detection')

run = st.checkbox('Start webcam')

FRAME_WINDOW = st.image([])

cap = None
if run:
    cap = cv2.VideoCapture(0)  # 0 is default webcam

    while run:
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to capture video")
            break

        # Preprocess and predict using detected face
        input_data = preprocess_face(frame)
        pred = model.predict(input_data)
        pred_class = np.argmax(pred, axis=1)[0]
        pred_label = reverse_class_indices[pred_class]

        # Optionally draw face rectangle
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) > 0:
            x, y, w, h = faces[0]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f'Ethnicity: {pred_label}', (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, f'Ethnicity: {pred_label}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)

        # Display frame (convert to RGB for Streamlit)
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
else:
    st.write('Webcam stopped')
