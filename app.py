import streamlit as st
from tensorflow import keras
from PIL import Image
import numpy as np
import pandas as pd

# ----------------------------------------------------
# PAGE CONFIGURATION
# ----------------------------------------------------

st.set_page_config(
    page_title="AI Histopathology Image Classifier",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------
# CUSTOM CSS
# ----------------------------------------------------

st.markdown("""
<style>

.main{
    background-color:#f5f7fa;
}

.title{
    text-align:center;
    font-size:42px;
    font-weight:bold;
    color:#004080;
}

.subtitle{
    text-align:center;
    font-size:18px;
    color:gray;
}

.result{
    background:#E8F5E9;
    padding:20px;
    border-radius:15px;
    border-left:8px solid green;
}

.info{
    background:#F8F9FA;
    padding:18px;
    border-radius:10px;
}

.warning{
    background:#FFF3CD;
    padding:15px;
    border-radius:10px;
    border-left:6px solid orange;
}

.footer{
    text-align:center;
    color:gray;
    font-size:14px;
}

</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# LOAD MODEL
# ----------------------------------------------------

@st.cache_resource
def load_model():
    return keras.models.load_model("oral_pathology_model_tf215.h5")

model = load_model()

# ----------------------------------------------------
# CLASS NAMES
# ----------------------------------------------------

class_names = [
    "AMELOBLASTOMA",
    "AOT"
]

# ----------------------------------------------------
# SIDEBAR
# ----------------------------------------------------

st.sidebar.title("🦷 Model Information")

st.sidebar.markdown("""
### About

This AI application classifies oral histopathological images into:

✅ Ameloblastoma

✅ Adenomatoid Odontogenic Tumour (AOT)

---

### Model

- Transfer Learning
- TensorFlow / Keras
- RGB Images
- Image Size: **224 × 224**

---

### Output

- Predicted Class
- Confidence Score
- Probability Distribution

---

### Clinical Notice

This AI model is intended for research,
education and demonstration purposes only.

It should **NOT** replace expert
histopathological diagnosis.
""")

# ----------------------------------------------------
# HEADER
# ----------------------------------------------------

st.markdown('<p class="title">🦷 AI Histopathology Image Classifier</p>',
unsafe_allow_html=True)

st.markdown(
'<p class="subtitle">Deep Learning-Based Classification of Oral Histopathology Images</p>',
unsafe_allow_html=True)

st.divider()

# ----------------------------------------------------
# TWO COLUMN LAYOUT
# ----------------------------------------------------

col1, col2 = st.columns([1,1])

with col1:

    st.subheader("📤 Upload Histopathology Image")

    uploaded_file = st.file_uploader(
        "Supported formats: JPG, JPEG, PNG",
        type=["jpg","jpeg","png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file)

        st.image(
            image,
            caption="Uploaded Histopathology Image",
            use_container_width=True
        )

with col2:

    st.subheader("📊 Prediction Results")

    if uploaded_file is not None:

        if st.button("🔍 Analyze Image", use_container_width=True):

            with st.spinner("Analyzing image..."):

                img = image.resize((224,224))

                img_array = np.array(img)

                img_array = np.expand_dims(img_array, axis=0)

                img_array = img_array / 255.0

                prediction = model.predict(img_array)

                confidence = float(np.max(prediction))

                predicted_class = class_names[np.argmax(prediction)]

            st.success("Analysis Complete")

            st.markdown(
            f"""
            <div class="result">

            <h2>Prediction</h2>

            <h1>{predicted_class}</h1>

            <h3>Confidence: {confidence:.2%}</h3>

            </div>
            """,
            unsafe_allow_html=True)

            st.write("")

            st.subheader("Confidence")

            st.progress(confidence)

            st.metric(
                label="Confidence Score",
                value=f"{confidence:.2%}"
            )

            st.write("")

            st.subheader("Prediction Probabilities")

            probs = prediction[0]

            df = pd.DataFrame(
                {
                    "Class": class_names,
                    "Probability": probs
                }
            ).set_index("Class")

            st.bar_chart(df)

            st.write("")

            st.markdown("""
            ### Clinical Interpretation

            The uploaded histopathology image is most consistent with the predicted lesion shown above.

            This prediction should always be interpreted alongside:

            - Clinical examination
            - Radiographic findings
            - Histopathological review
            - Specialist opinion
            """)

            with st.expander("🔬 AI Model Details"):

                st.write("Model: TensorFlow / Keras")

                st.write("Architecture: Transfer Learning")

                st.write("Input Size: 224 × 224")

                st.write("Color Channels: RGB")

                st.write("Normalization: Pixel values divided by 255")

                st.write("Classes:")

                for c in class_names:
                    st.write("-", c)

# ----------------------------------------------------
# DISCLAIMER
# ----------------------------------------------------

st.write("")

st.markdown("""
<div class="warning">

### ⚠ Disclaimer

This application is intended for educational,
research and demonstration purposes only.

Predictions generated by this AI system should
not be used as the sole basis for diagnosis,
clinical decision-making or treatment planning.

All cases should be reviewed by qualified oral
pathologists and healthcare professionals.

</div>
""",
unsafe_allow_html=True)

# ----------------------------------------------------
# FOOTER
# ----------------------------------------------------

st.write("")
st.divider()

st.markdown("""
<div class="footer">

Developed by <b>Dr. Akinshipo & Dr. Udebuana </b><br>

RobotProf AI • TensorFlow • Streamlit • Deep Learning

© 2026

</div>
""",
unsafe_allow_html=True)
