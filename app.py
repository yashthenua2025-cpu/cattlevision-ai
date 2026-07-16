import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="CattleVision AI",
    page_icon="🐄",
    layout="centered"
)


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "cow_buffalo_classifier.keras"
    )


model = load_model()

class_names = ["Cow", "Buffalo"]

st.title("🐄 CattleVision AI")
st.subheader("AI-Based Cow & Buffalo Classification")

st.write(
    "Upload an image of a cow or buffalo and let the AI classify it."
)

uploaded_file = st.file_uploader(
    "📸 Upload Animal Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("🔍 Predict Animal"):

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

            predicted_class = np.argmax(prediction[0])

            confidence = np.max(
                prediction[0]
            ) * 100

            animal = class_names[predicted_class]

        st.success("Prediction Complete!")

        st.markdown("---")

        st.subheader("🎯 Prediction Result")

        if animal == "Cow":
            st.success(
                f"🐄 This is a {animal}"
            )
        else:
            st.info(
                f"🐃 This is a {animal}"
            )

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )