import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

# Set the background color of the app
st.markdown(
    """
    <style>
        body {
            background-color: violet;
        }
        .title {
            text-align: center;
            color: white;
            font-size: 36px;
        }
    </style>
    """, unsafe_allow_html=True)

# Title of the app
st.markdown('<h1 class="title">Do you have Oral Cancer?</h1>', unsafe_allow_html=True)

# Load YOLOv8 model
model = YOLO("model = YOLO("best.pt")   # Update with your correct model path

# Image upload or camera option
option = st.radio("Choose image source:", ("Upload Image", "Use Camera"))

# Initialize image variable
image = None

# Upload image option
if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

# Camera capture option
elif option == "Use Camera":
    camera_image = st.camera_input("Take a picture")
    if camera_image is not None:
        image = Image.open(camera_image)
        st.image(image, caption="Captured Image", use_column_width=True)

# Perform predictions if image is available
if image is not None:
    # Perform object detection
    results = model.predict(image)

    # Initialize flags for cancer detection
    has_multinucleated = False
    has_larger_than_normal = False

    # Check detected classes
    for box in results[0].boxes:
        class_name = model.names[int(box.cls)]
        if class_name == "Multinucleated":
            has_multinucleated = True
        elif class_name == "Larger than Normal":
            has_larger_than_normal = True

    # Show annotated image
    annotated_image = results[0].plot()
    st.image(annotated_image, caption="Detected Cells", use_column_width=True)

    # Conclusion
    st.subheader("Conclusion:")
    if has_multinucleated or has_larger_than_normal:
        st.markdown("⚠️ **Thus, you have oral cancer.**")
    else:
        st.markdown("✅ **Thus, you don’t have any oral cancer.**")
