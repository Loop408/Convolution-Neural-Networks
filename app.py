import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

# Page config
st.set_page_config(
    page_title="Pneumonia Detection",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E86AB;
        font-size: 2.5em;
        margin-bottom: 0.5em;
    }
    .sub-header {
        text-align: center;
        color: #A23B72;
        font-size: 1.2em;
        margin-bottom: 2em;
    }
    .upload-section {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .result-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Load trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("pneumonia_model.h5")
    return model

model = load_model()

# Sidebar for inputs
with st.sidebar:
    st.title("🩺 Upload & Analyze")
    st.markdown("---")
    uploaded_file = st.file_uploader(
        "Choose a Chest X-ray Image",
        type=["jpg", "png", "jpeg"],
        help="Upload a clear chest X-ray image for analysis"
    )

    if uploaded_file is not None:
        analyze_button = st.button("🔍 Analyze Image", type="primary", use_container_width=True)
    else:
        analyze_button = False

    st.markdown("---")
    st.markdown("### Instructions")
    st.markdown("""
    1. Upload a chest X-ray image
    2. Click 'Analyze Image' to get results
    3. Review the prediction and probabilities
    """)

# Main content
st.markdown('<h1 class="main-header">Pneumonia Detection from Chest X-Ray</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Medical Image Analysis Tool</p>', unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center; margin-bottom: 30px;">
    Upload a chest X-ray image to analyze the risk of pneumonia using our advanced AI model.
    Get instant results with confidence scores and risk assessment.
</div>
""", unsafe_allow_html=True)

# Initialize session state for results
if 'results' not in st.session_state:
    st.session_state.results = None

# Image preview and analysis
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📷 Original Image")
        st.image(image, use_column_width=True, caption="Uploaded X-ray")

    # Resize for model
    resized = image.resize((150, 150))

    with col2:
        st.subheader("🔧 Processed Image")
        st.image(resized, use_column_width=True, caption="Model Input (150x150)")

    # Analysis section
    if analyze_button:
        with st.spinner("Analyzing image... Please wait."):
            # Convert to array
            img_array = np.array(resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array)[0][0]

            pneumonia_prob = float(prediction)
            normal_prob = 1 - pneumonia_prob

            # Store results in session state
            st.session_state.results = {
                'pneumonia_prob': pneumonia_prob,
                'normal_prob': normal_prob
            }

# Display results if available
if st.session_state.results is not None:
    pneumonia_prob = st.session_state.results['pneumonia_prob']
    normal_prob = st.session_state.results['normal_prob']

    st.markdown("---")
    st.header("📊 Analysis Results")

    # Main result
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.subheader("🔍 Prediction")

        if pneumonia_prob > 0.5:
            st.error(f"**PNEUMONIA DETECTED**")
            st.metric("Confidence", f"{pneumonia_prob*100:.1f}%")
        else:
            st.success(f"**NORMAL**")
            st.metric("Confidence", f"{normal_prob*100:.1f}%")

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.subheader("⚠️ Risk Level")

        if pneumonia_prob > 0.8:
            st.error("HIGH RISK")
            risk_color = "🔴"
        elif pneumonia_prob > 0.5:
            st.warning("MODERATE RISK")
            risk_color = "🟡"
        else:
            st.success("LOW RISK")
            risk_color = "🟢"

        st.markdown(f"**{risk_color} { 'HIGH' if pneumonia_prob > 0.8 else 'MODERATE' if pneumonia_prob > 0.5 else 'LOW' }**")
        st.markdown('</div>', unsafe_allow_html=True)

    # Detailed probabilities
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.subheader("📈 Probability Distribution")

    prob_data = {
        "Normal": normal_prob,
        "Pneumonia": pneumonia_prob
    }

    st.bar_chart(prob_data, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Normal Probability", f"{normal_prob*100:.1f}%")
    with col2:
        st.metric("Pneumonia Probability", f"{pneumonia_prob*100:.1f}%")

    st.markdown('</div>', unsafe_allow_html=True)

    # Reset button
    if st.button("🔄 Analyze Another Image", use_container_width=True):
        st.session_state.results = None
        st.rerun()