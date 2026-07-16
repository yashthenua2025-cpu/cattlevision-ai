import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="CattleVision AI",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #050505 0%, #101010 50%, #050505 100%);
    color: white;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}

.hero {
    padding: 45px 35px;
    border-radius: 28px;
    background: linear-gradient(135deg, #171717, #080808);
    border: 1px solid #292929;
    box-shadow: 0 20px 60px rgba(0,0,0,0.45);
    margin-bottom: 30px;
}

.badge {
    display: inline-block;
    padding: 8px 16px;
    border-radius: 30px;
    background: rgba(34, 197, 94, 0.12);
    border: 1px solid rgba(34, 197, 94, 0.3);
    color: #4ade80;
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.5px;
}

.hero h1 {
    font-size: 54px;
    font-weight: 800;
    margin: 20px 0 10px 0;
    background: linear-gradient(90deg, #ffffff, #9ca3af);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero p {
    color: #9ca3af;
    font-size: 17px;
    max-width: 650px;
    line-height: 1.7;
}

.section-title {
    font-size: 24px;
    font-weight: 700;
    margin: 20px 0 15px 0;
}

.info-card {
    padding: 22px;
    border-radius: 20px;
    background: #111111;
    border: 1px solid #262626;
    min-height: 145px;
}

.info-card h3 {
    margin: 0 0 10px 0;
    color: white;
}

.info-card p {
    color: #9ca3af;
    font-size: 14px;
    line-height: 1.6;
}

.result-card {
    padding: 35px;
    border-radius: 25px;
    background: linear-gradient(135deg, #171717, #0b0b0b);
    border: 1px solid #303030;
    text-align: center;
    margin-top: 25px;
    box-shadow: 0 15px 45px rgba(0,0,0,0.35);
}

.result-animal {
    font-size: 42px;
    font-weight: 800;
    margin: 10px 0;
}

.confidence {
    font-size: 18px;
    color: #4ade80;
    font-weight: 600;
}

.footer {
    text-align: center;
    color: #6b7280;
    font-size: 13px;
    margin-top: 50px;
    padding-top: 20px;
    border-top: 1px solid #242424;
}

div.stButton > button {
    width: 100%;
    border-radius: 14px;
    padding: 14px;
    background: linear-gradient(90deg, #22c55e, #16a34a);
    color: white;
    font-weight: 700;
    border: none;
    font-size: 16px;
}

div.stButton > button:hover {
    background: linear-gradient(90deg, #16a34a, #15803d);
    color: white;
}

[data-testid="stFileUploader"] {
    background: #111111;
    border-radius: 20px;
    border: 1px dashed #404040;
    padding: 15px;
}

</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "cow_buffalo_classifier.keras"
    )


model = load_model()

class_names = ["Cow", "Buffalo"]


st.markdown("""
<div class="hero">
    <span class="badge">● AI MODEL ONLINE</span>
    <h1>CattleVision AI</h1>
    <p>
        Intelligent cattle image classification powered by deep learning.
        Upload a cattle image and let our AI identify whether it is a cow or buffalo.
    </p>
</div>
""", unsafe_allow_html=True)


st.markdown(
    '<div class="section-title">AI Classification System</div>',
    unsafe_allow_html=True
)


col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="info-card">
        <h3>🧠 Deep Learning</h3>
        <p>AI-powered image classification model trained to identify cattle types.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="info-card">
        <h3>⚡ Fast Prediction</h3>
        <p>Get classification results within seconds after uploading an image.</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="info-card">
        <h3>🎯 Confidence Score</h3>
        <p>View the AI model's confidence level for every prediction.</p>
    </div>
    """, unsafe_allow_html=True)


st.markdown(
    '<div class="section-title">Upload Animal Image</div>',
    unsafe_allow_html=True
)


uploaded_file = st.file_uploader(
    "Upload a JPG, JPEG or PNG image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

    with col2:

        st.markdown(
            "### Ready for AI Analysis"
        )

        st.write(
            "Click the button below to classify the uploaded cattle image."
        )

        predict_button = st.button(
            "🔍 ANALYZE IMAGE"
        )

        if predict_button:

            with st.spinner("AI is analyzing the image..."):

                img = image.resize((224, 224))

                img_array = np.array(img)

                img_array = np.expand_dims(
                    img_array,
                    axis=0
                )

                prediction = model.predict(
                    img_array,
                    verbose=0
                )

                predicted_class = np.argmax(
                    prediction[0]
                )

                confidence = np.max(
                    prediction[0]
                ) * 100

                animal = class_names[predicted_class]

            emoji = "🐄" if animal == "Cow" else "🐃"

            st.markdown(f"""
            <div class="result-card">
                <div style="font-size: 55px;">{emoji}</div>
                <div style="color:#9ca3af;">AI PREDICTION</div>
                <div class="result-animal">{animal}</div>
                <div class="confidence">
                    Confidence: {confidence:.2f}%
                </div>
            </div>
            """, unsafe_allow_html=True)


st.markdown("""
<div class="footer">
    CattleVision AI • Deep Learning Based Animal Classification
    <br>
    Built for AI & Computer Vision Research
</div>
""", unsafe_allow_html=True)
