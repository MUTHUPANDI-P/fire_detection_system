import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Set up the Streamlit app
st.set_page_config(page_title="Fire Detection App", layout="centered")
st.title(" 🔥 Fire Detection App")
st.markdown("Upload one or more images to detect fire.")

# Load the trained model
@st.cache_resource
def load_fire_model():
    return load_model("fire_cnn_model.keras")

model = load_fire_model()

# File uploader
uploaded_files = st.file_uploader("Choose Image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# Prediction function
def predict_fire(img_file):
    img = Image.open(img_file).convert("RGB").resize((128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    confidence = float(prediction[0][0])  # fire probability
    return confidence

# Display results
if uploaded_files:
    for uploaded_file in uploaded_files:
        st.image(uploaded_file, caption="Uploaded Image", width=250)
        confidence = predict_fire(uploaded_file)

        is_fire = confidence < 0.5
        fire_prob = confidence if is_fire else 1 - confidence
        no_fire_prob = 1 - fire_prob

        # Result message
        if is_fire:
            st.markdown(f"### 🔥 Fire Detected ")
        else:
            st.markdown(f"### ✅ No Fire Detected ")
