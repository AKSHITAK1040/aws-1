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

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def preprocess_face(frame, img_size=(128,128)):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) > 0:
        x, y, w, h = faces[0]
        face = frame[y:y+h, x:x+w]
    else:
        face = frame
    img = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, img_size)
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)

# Page configuration
st.set_page_config(page_title="Ethnicity Predictor", layout="wide")

# Custom UI with title and description
st.markdown("<h1 style='text-align: center; color: #4B0082;'>Ethnicity Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #6A5ACD;'>Predict your ethnicity using your webcam</h4>", unsafe_allow_html=True)
st.write("---")

# Create columns for camera and prediction
col1, col2 = st.columns([1,1])

with col1:
    st.subheader("📸 Capture Your Face")
    uploaded_image = st.camera_input("Take a picture")
    
with col2:
    st.subheader("📝 Prediction Result")
    result_container = st.empty()  # placeholder for prediction results

if uploaded_image is not None:
    # Convert PIL RGB image to OpenCV BGR
    image = np.array(Image.open(uploaded_image))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Preprocess and predict
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
        cv2.putText(image, f'{pred_label}', (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(image, f'{pred_label}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the image with prediction
    col1.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_column_width=True)

    # Show prediction info
    with col2:
        st.success(f"Predicted Ethnicity: **{pred_label}**")
        st.progress(min(int(np.max(pred)*100), 100))  # show confidence as progress bar
        st.write("Confidence Scores:")
        for idx, score in enumerate(pred[0]):
            st.write(f"- {reverse_class_indices[idx]}: {score:.2f}")
