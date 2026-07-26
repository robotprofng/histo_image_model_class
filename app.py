
import streamlit as st
from tensorflow import keras
from PIL import Image
import numpy as np


@st.cache_resource
def load_model():
    return keras.models.load_model(
        "oral_pathology_model_tf215.h5",
        compile=False
    )

model = load_model()

class_names = [
    "AMELOBLASTOMA",
    "AOT"
]


st.title("AMELOBLASTOMA VS AOT AI IMAGE CLASSIFIER")

st.write(
    "AI-assisted classification of oral histopathology images. Upload an image to obtain a predicted lesion category.")



uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded image"
    )


    img = image.resize((224,224))

    img_array = np.array(img)

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    img_array = img_array / 255.0


    prediction = model.predict(img_array)


    predicted_class = class_names[
        np.argmax(prediction)
    ]

    confidence = np.max(prediction)


    st.success(
        f"Prediction: {predicted_class}"
    )

    st.write(
        f"Confidence: {confidence:.2%}"
    )
