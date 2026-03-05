import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="Pneumonia Detection", layout="centered")

# Load trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("pneumonia_model.h5")
    return model

model = load_model()

# -------------------------------
# 1️⃣ HEADER SECTION
# -------------------------------

st.title("🩺 Pneumonia Detection from Chest X-Ray")
st.subheader("AI Assisted Screening Tool")

st.write(
"""
Upload a **Chest X-ray image** to check the risk of pneumonia using an AI model.
"""
)

# -------------------------------
# 2️⃣ FILE UPLOAD SECTION
# -------------------------------

uploaded_file = st.file_uploader(
    "Upload X-ray Image",
    type=["jpg", "png"]
)

# -------------------------------
# 3️⃣ IMAGE PREVIEW
# -------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.write("### Uploaded Image")
        st.image(image, use_column_width=True)

    # Resize for model
    resized = image.resize((150,150))

    with col2:
        st.write("### Resized Image (Model Input)")
        st.image(resized, use_column_width=True)

    # Convert to array
    img_array = np.array(resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

# -------------------------------
# 4️⃣ PREDICTION BUTTON
# -------------------------------

    if st.button("🔍 Analyze Image"):

        prediction = model.predict(img_array)[0][0]

        pneumonia_prob = float(prediction)
        normal_prob = 1 - pneumonia_prob

        # -------------------------------
        # A. PREDICTION RESULT CARD
        # -------------------------------

        st.markdown("---")

        st.subheader("📊 Prediction Result")

        if pneumonia_prob > 0.5:

            st.error(
                f"""
                **Status: PNEUMONIA**

                **Confidence:** {pneumonia_prob*100:.2f}%
                """
            )

        else:

            st.success(
                f"""
                **Status: NORMAL**

                **Confidence:** {normal_prob*100:.2f}%
                """
            )

        # -------------------------------
        # B. PROBABILITY DISPLAY
        # -------------------------------

        st.write("### Probability Distribution")

        prob_data = {
            "Normal": normal_prob,
            "Pneumonia": pneumonia_prob
        }

        st.bar_chart(prob_data)

        st.write(f"Normal: {normal_prob*100:.2f}%")
        st.write(f"Pneumonia: {pneumonia_prob*100:.2f}%")

        # -------------------------------
        # C. RISK INDICATOR
        # -------------------------------

        if pneumonia_prob > 0.8:
            risk = "HIGH"
            st.error("⚠️ Risk Level: HIGH")

        elif pneumonia_prob > 0.5:
            risk = "MODERATE"
            st.warning("⚠️ Risk Level: MODERATE")

        else:
            risk = "LOW"
            st.success("✅ Risk Level: LOW")

        # -------------------------------
        # D. DISCLAIMER
        # -------------------------------

        st.markdown("---")

        st.warning(
        """
        ⚠️ **Disclaimer**

        This AI tool is for **educational purposes only**.  
        It should **not be used as a medical diagnosis tool**.

        Please consult a **certified radiologist or doctor** for medical advice.
        """
        )