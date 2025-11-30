# streamlit_yolo_app.py

import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

# -------------------------------
# Streamlit App Title
# -------------------------------
st.title("YOLOv8 Object Detection")
model = YOLO("yolov8n.pt")

# -------------------------------
# Load YOLO Model
@st.cache_resource
def load_model(model_name):
    return YOLO(model_name)

# -------------------------------
# Upload Image
# -------------------------------
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    # Perform Inference
    results = model(img_array)

    # Annotated results
    annotated_img = results[0].plot()

    # Get shape of annotated image
    h, w, c = annotated_img.shape
    st.write(f"**Annotated Image Shape:** {h} x {w} x {c}")

    # Display Input and Output
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original Image", use_column_width=True)
    with col2:
        st.image(annotated_img, caption="Detected Objects", use_column_width=True)

    # Option to download the annotated image
    result_bgr = cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite("annotated_output.jpg", result_bgr)
    with open("annotated_output.jpg", "rb") as f:
        st.download_button("Download Annotated Image", f, "annotated_output.jpg")