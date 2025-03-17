import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

# Set the background color of the app and styling elements
st.markdown(
    """
    <style>
        body {
            background-color: violet;
            font-family: Arial, sans-serif;
        }
        .title {
            text-align: center;
            color: white;
            font-size: 40px;
            padding: 20px;
            background-color: #6a0dad;
            border-radius: 10px;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.2);
        }
        .sidebar {
            background-color: #9b59b6;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        }
        .button {
            background-color: #00BFFF;
            color: white;
            border-radius: 5px;
            padding: 10px;
            font-size: 16px;
            cursor: pointer;
            border: none;
        }
        .button:hover {
            background-color: #009ACD;
        }
        .container {
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
            gap: 20px;
            padding: 20px;
        }
        .image-container {
            border: 2px solid #6a0dad;
            border-radius: 10px;
            padding: 10px;
            background-color: white;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
            max-width: 100%;
        }
        .conclusion {
            background-color: #f0f0f0;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        }
    </style>
    """, unsafe_allow_html=True
)

# Title of the app
st.markdown('<h1 class="title">Do you have Oral Cancer?</h1>', unsafe_allow_html=True)

# Load YOLOv8 model
model = YOLO("C:/Users/Lab Telecom/Documents/APP/best.pt")  # Update with your correct model path

# Create a sidebar with options
st.sidebar.markdown('<div class="sidebar"><h2 style="color:white;">Options</h2></div>', unsafe_allow_html=True)

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

    # Show annotated image inside a styled box
    annotated_image = results[0].plot()
    st.markdown('<div class="image-container">', unsafe_allow_html=True)
    st.image(annotated_image, caption="Detected Cells", use_column_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Conclusion section inside a styled box
    st.markdown('<div class="conclusion">', unsafe_allow_html=True)
    st.subheader("Conclusion:")
    if has_multinucleated or has_larger_than_normal:
        st.markdown("⚠️ **Thus, you have oral cancer.**")
        st.markdown("### Follow these recommendations for treatment and support:")

        st.markdown("1. **Follow Medical Treatment**")
        st.markdown("   - Consult with an oncologist and a dentist to develop a treatment plan.")
        st.markdown("   - Follow recommended treatments such as surgery, radiation therapy, chemotherapy, or targeted therapy.")
        st.markdown("   - Attend all scheduled medical checkups to monitor progress.")

        st.markdown("2. **Improve Diet and Nutrition**")
        st.markdown("   - Eat a balanced diet rich in vitamins and antioxidants to support healing.")
        st.markdown("   - Avoid spicy, acidic, and very hot foods that may irritate the mouth.")
        st.markdown("   - Stay hydrated and consider soft foods if chewing/swallowing is painful.")

        st.markdown("3. **Maintain Good Oral Hygiene**")
        st.markdown("   - Use a soft-bristled toothbrush and a mild, non-alcoholic mouthwash.")
        st.markdown("   - Rinse your mouth with warm salt water to prevent infections.")
        st.markdown("   - Visit a dentist regularly to manage oral health during treatment.")

        st.markdown("4. **Avoid Harmful Substances**")
        st.markdown("   - Quit smoking and alcohol consumption, as they can worsen the condition and increase recurrence risk.")
        st.markdown("   - Avoid chewing tobacco or betel nuts, as they are linked to oral cancer.")

        st.markdown("5. **Seek Emotional Support**")
        st.markdown("   - Join a support group or talk to a counselor to cope with emotional stress.")
        st.markdown("   - Engage in stress-relieving activities such as meditation, yoga, or hobbies.")

    else:
        st.markdown("✅ **Thus, you don’t have any oral cancer.**")
        st.markdown("### Prevention Tips to Stay Safe:")
        
        st.markdown("1. **Avoid Tobacco and Alcohol**")
        st.markdown("   - Do not smoke or use tobacco products (cigarettes, cigars, chewing tobacco, betel nuts).")
        st.markdown("   - Limit or avoid alcohol consumption, as it increases cancer risk.")

        st.markdown("2. **Maintain Good Oral Hygiene**")
        st.markdown("   - Brush and floss daily to keep gums and teeth healthy.")
        st.markdown("   - Get regular dental checkups to detect early signs of oral health problems.")

        st.markdown("3. **Eat a Healthy Diet**")
        st.markdown("   - Consume fruits and vegetables rich in antioxidants, such as carrots, leafy greens, and citrus fruits.")
        st.markdown("   - Reduce processed and red meat consumption, as they may increase cancer risks.")

        st.markdown("4. **Protect Yourself from HPV (Human Papillomavirus)**")
        st.markdown("   - HPV infection is linked to oral cancer. Getting the HPV vaccine can help prevent it.")
        st.markdown("   - Practice safe oral hygiene and safe sex to reduce the risk of HPV transmission.")

        st.markdown("5. **Be Aware of Early Symptoms**")
        st.markdown("   - Watch for persistent mouth sores, lumps, white/red patches, difficulty swallowing, or unexplained pain in the mouth.")
        st.markdown("   - If you notice any unusual changes in your mouth, consult a doctor or dentist immediately.")
    st.markdown('</div>', unsafe_allow_html=True)
