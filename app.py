# ============================================================
# HistoAI - Oral Pathology AI Classifier
# Developed by Dr. Irene Udebuana | RobotProf AI
# ============================================================


import streamlit as st
from tensorflow import keras
from PIL import Image
import numpy as np
import pandas as pd
import time


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="HistoAI | Oral Pathology AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# DARK BLUE PREMIUM UI
# ============================================================


st.markdown("""

<style>


/* MAIN BACKGROUND */

.stApp {

background:
linear-gradient(
135deg,
#020617,
#0f172a,
#172554
);

color:white;

}


/* ALL TEXT */

html, body, [class*="css"] {

color:white;

font-family:
"Inter",
Arial,
sans-serif;

}



/* SIDEBAR */

section[data-testid="stSidebar"] {

background:

linear-gradient(
180deg,
#020617,
#0b1f44
);

}



/* HEADINGS */

h1,h2,h3,h4 {

color:white !important;

}



/* HERO */

.hero {

background:

linear-gradient(
135deg,
#1e40af,
#0f766e
);

padding:45px;

border-radius:25px;

text-align:center;

box-shadow:
0 20px 50px rgba(0,0,0,0.5);

margin-bottom:35px;

}


.hero h1 {

font-size:48px;

font-weight:800;

}


.hero p {

font-size:20px;

color:#e0f2fe;

}




/* CARDS */

.card {

background:

rgba(255,255,255,0.08);

backdrop-filter:blur(15px);

padding:30px;

border-radius:25px;

border:

1px solid rgba(255,255,255,0.15);

box-shadow:

0 15px 40px rgba(0,0,0,0.4);

}




/* METRIC CARDS */

.metric {

background:

rgba(255,255,255,0.1);

padding:25px;

border-radius:20px;

text-align:center;

border:

1px solid rgba(255,255,255,0.2);

}


.metric h2 {

font-size:35px;

}



.metric p {

color:#cbd5e1;

}



/* RESULT */

.success-box {

background:

linear-gradient(
135deg,
#065f46,
#047857
);

padding:30px;

border-radius:25px;

text-align:center;

box-shadow:
0 10px 30px rgba(0,0,0,0.4);

}



.success-box h1 {

font-size:40px;

}



/* WARNING */

.warning {

background:

rgba(245,158,11,0.15);

border-left:

8px solid #f59e0b;

padding:25px;

border-radius:20px;

}



/* BUTTON */

.stButton button {


background:

linear-gradient(
90deg,
#2563eb,
#06b6d4
);


color:white;

font-weight:bold;

font-size:18px;

height:55px;

border-radius:15px;

border:none;

width:100%;


}



.stButton button:hover {

transform:scale(1.03);

}




/* UPLOADER */

[data-testid="stFileUploader"] {

background:
rgba(255,255,255,0.05);

padding:15px;

border-radius:20px;

}



.footer {

text-align:center;

color:#94a3b8;

font-size:14px;

}



</style>


""", unsafe_allow_html=True)




# ============================================================
# LOAD MODEL
# ============================================================


@st.cache_resource

def load_model():

    model = keras.models.load_model(
        "oral_pathology_model_tf215.h5"
    )

    return model



model = load_model()




# ============================================================
# CLASS LABELS
# ============================================================


classes = [

"Ameloblastoma",

"Adenomatoid Odontogenic Tumour (AOT)"

]




# ============================================================
# SIDEBAR
# ============================================================


with st.sidebar:


    st.markdown(

    """

    # 🔬 HistoAI

    ## Oral Pathology AI Assistant


    ---

    ### MODEL

    🧠 Deep Learning CNN

    ⚙ TensorFlow/Keras

    📌 Transfer Learning


    ### IMAGE INPUT

    🖼 Histopathology Images

    📐 224 × 224 RGB


    ### CLASSIFICATION

    🔬 Ameloblastoma

    🔬 AOT


    ---

    Developed by:

    **Dr. Irene Udebuana**

    RobotProf AI


    """

    )





# ============================================================
# HERO SECTION
# ============================================================



st.markdown(

"""

<div class="hero">


<h1>
🔬 HistoAI Oral Pathology Classifier
</h1>


<p>
Artificial Intelligence for Histopathological Image Classification
</p>


</div>


""",

unsafe_allow_html=True

)




# ============================================================
# DASHBOARD METRICS
# ============================================================



a,b,c,d = st.columns(4)



metrics=[

("🧠","AI MODEL","CNN"),

("📷","INPUT","Histology"),

("🎯","TASK","Classification"),

("⚡","MODE","Real-time")

]


for col,data in zip(
[a,b,c,d],
metrics
):

    with col:

        st.markdown(

        f"""

        <div class="metric">

        <h2>{data[0]}</h2>

        <b>{data[1]}</b>

        <p>{data[2]}</p>


        </div>

        """,

        unsafe_allow_html=True

        )




st.write("")



# ============================================================
# MAIN APPLICATION
# ============================================================


left,right = st.columns(
[1,1]
)




with left:


    st.markdown(
    '<div class="card">',
    unsafe_allow_html=True
    )


    st.subheader(
    "📤 Upload Histopathology Image"
    )


    uploaded = st.file_uploader(

    "Select image",

    type=[
        "png",
        "jpg",
        "jpeg"
    ]

    )



    if uploaded:


        image = Image.open(
            uploaded
        )


        st.image(

        image,

        use_container_width=True,

        caption="Uploaded Histology Image"

        )


    st.markdown(
    "</div>",
    unsafe_allow_html=True
    )





with right:


    st.markdown(
    '<div class="card">',
    unsafe_allow_html=True
    )


    st.subheader(
    "🤖 AI Prediction"
    )


    if uploaded:


        if st.button(
        "🚀 Analyze Image"
        ):


            with st.spinner(
            "AI is examining microscopic patterns..."
            ):


                time.sleep(2)



                img=image.resize(
                (224,224)
                )


                img=np.array(
                img
                )


                img=np.expand_dims(
                img,
                axis=0
                )


                img=img/255.0



                prediction=model.predict(
                img
                )



                confidence=float(
                np.max(prediction)
                )


                index=np.argmax(
                prediction
                )


                result=classes[index]



            st.markdown(

            f"""

            <div class="success-box">

            <h2>
            AI Prediction
            </h2>


            <h1>
            {result}
            </h1>


            <h2>

            Confidence:

            {confidence:.2%}

            </h2>


            </div>

            """,

            unsafe_allow_html=True

            )


            st.write("")


            st.subheader(
            "Confidence Level"
            )


            st.progress(
            confidence
            )


            df=pd.DataFrame(

            {

            "Class":
            classes,

            "Probability":
            prediction[0]

            }

            )


            st.subheader(
            "Prediction Distribution"
            )


            st.bar_chart(

            df.set_index(
            "Class"
            )

            )




    else:


        st.info(
        "Upload a histology image to start AI analysis."
        )



    st.markdown(
    "</div>",
    unsafe_allow_html=True
    )





# ============================================================
# MODEL INFORMATION
# ============================================================


st.write("")



with st.expander(
"🔬 View AI Architecture"
):


    st.write(

    """

    **Model:** Oral Pathology Deep Learning Classifier


    **Framework:**

    TensorFlow / Keras


    **Input:**

    224 × 224 RGB histopathology images


    **Processing:**

    - Image resizing

    - Normalization

    - CNN feature extraction

    - Binary classification


    **Predicted lesions:**

    - Ameloblastoma

    - Adenomatoid Odontogenic Tumour


    """

    )





# ============================================================
# DISCLAIMER
# ============================================================


st.markdown(

"""

<div class="warning">


<h3>
⚠ Medical Disclaimer
</h3>


This AI tool is designed for:

✔ Research

✔ Education

✔ Demonstration


It does not replace:

- Histopathological diagnosis

- Specialist review

- Clinical judgement


Final diagnosis must be performed by qualified healthcare professionals.


</div>


""",

unsafe_allow_html=True

)




# ============================================================
# FOOTER
# ============================================================


st.divider()


st.markdown(

"""

<div class="footer">


Developed by <b>Dr. Irene Udebuana</b>


<br>

RobotProf AI Healthcare Education


<br><br>


TensorFlow • Streamlit • Deep Learning


<br>

© 2026


</div>


""",

unsafe_allow_html=True

)

