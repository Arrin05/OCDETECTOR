import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Custom CSS for colorful styling
st.markdown(
    """
    <style>
        body {
            background: linear-gradient(135deg, #ff9a9e, #fad0c4);
            font-family: 'Arial', sans-serif;
        }
        .title {
            text-align: center;
            color: white;
            font-size: 45px;
            padding: 20px;
            background: linear-gradient(90deg, #6a11cb, #2575fc);
            border-radius: 15px;
            box-shadow: 0px 5px 10px rgba(0, 0, 0, 0.3);
        }
        .sidebar {
            background: linear-gradient(135deg, #ff758c, #ff7eb3);
            padding: 20px;
            border-radius: 15px;
            color: white;
        }
        .image-container, .conclusion {
            background: white;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);
        }
        .how-to-use {
            background: #ff9a9e;
            padding: 15px;
            border-radius: 10px;
            color: white;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# App Title
st.markdown('<h1 class="title">Do you have Oral Cancer?</h1>', unsafe_allow_html=True)

# Load YOLO model
model = YOLO("C:/Users/Lab Telecom/Documents/APP/best.pt")

# Sidebar with options
st.sidebar.markdown('<div class="sidebar"><h2>Options</h2></div>', unsafe_allow_html=True)
option = st.radio("Choose image source:", ("Upload Image", "Use Camera"))

# Toggle for How to Use
if st.toggle("How to Use?"):
    st.markdown('<div class="how-to-use"><h3>HOW TO USE LAWAI?</h3></div>', unsafe_allow_html=True)
    if option == "Upload Image":
        st.warning("**Using Upload:** Upload a picture containing a MICROSCOPIC VIEW OF EPITHELIAL CELLS STRICTLY.")
    else:
        st.warning("**Using Camera:** You MUST use APEXEL hardware to see the microscopic view of epithelial cells in saliva.")

# Upload or capture image
image = None
if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
elif option == "Use Camera":
    camera_image = st.camera_input("Take a picture")
    if camera_image:
        image = Image.open(camera_image)
        st.image(image, caption="Captured Image", use_column_width=True)

# Process image if available
if image:
    results = model.predict(image)
    has_cancer = any(model.names[int(box.cls)] in ["Multinucleated", "Larger than Normal"] for box in results[0].boxes)
    annotated_image = results[0].plot()
    
    st.markdown('<div class="image-container">', unsafe_allow_html=True)
    st.image(annotated_image, caption="Detected Cells", use_column_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Conclusion Section
    st.markdown('<div class="conclusion">', unsafe_allow_html=True)
    st.subheader("Conclusion:")
    if has_cancer:
        st.error("⚠️ **You have oral cancer.**")
        st.markdown("### Follow these recommendations for treatment and support:")
        st.markdown("1. **Follow Medical Treatment**")
        st.markdown("   - Consult with an oncologist and dentist.")
        st.markdown("   - Follow prescribed treatments like surgery, chemotherapy, or radiation.")
        st.markdown("   - Regularly attend medical checkups.")
        
        st.markdown("2. **Improve Diet and Nutrition**")
        st.markdown("   - Eat a balanced diet rich in vitamins and antioxidants.")
        st.markdown("   - Avoid spicy and acidic foods.")
        
        st.markdown("3. **Maintain Good Oral Hygiene**")
        st.markdown("   - Use a soft-bristled toothbrush and mild mouthwash.")
        st.markdown("   - Rinse with warm salt water to prevent infections.")
        
        st.markdown("4. **Avoid Harmful Substances**")
        st.markdown("   - Quit smoking and alcohol consumption.")
        st.markdown("   - Avoid chewing tobacco and betel nuts.")
        
        st.markdown("5. **Seek Emotional Support**")
        st.markdown("   - Join a support group or speak to a counselor.")
        
    else:
        st.success("✅ **You don’t have oral cancer.**")
        st.markdown("### Prevention Tips:")
        st.markdown("1. **Avoid Tobacco and Alcohol**")
        st.markdown("   - Do not smoke or use tobacco products.")
        st.markdown("   - Limit alcohol consumption.")
        
        st.markdown("2. **Maintain Good Oral Hygiene**")
        st.markdown("   - Brush and floss daily.")
        st.markdown("   - Regular dental checkups are essential.")
        
        st.markdown("3. **Eat a Healthy Diet**")
        st.markdown("   - Consume fruits and vegetables rich in antioxidants.")
        st.markdown("   - Reduce processed and red meat consumption.")
        
        st.markdown("4. **Be Aware of Early Symptoms**")
        st.markdown("   - Look for persistent mouth sores, lumps, or pain.")
        st.markdown("   - Consult a doctor if unusual symptoms appear.")
    
    st.markdown('</div>', unsafe_allow_html=True)
