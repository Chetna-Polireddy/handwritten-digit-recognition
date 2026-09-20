import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# Load trained model
model = tf.keras.models.load_model("handwritten_digit_model.keras")

# Page configuration
st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="🔢",
    layout="centered"
)

# Title
st.title("🔢 Handwritten Digit Recognition")
st.write("Draw a digit from 0 to 9 and let the neural network predict it!")

# Clear button
if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0

if st.button("🧹 Clear Canvas"):
    st.session_state.canvas_key += 1
    st.rerun()

# Drawing canvas
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=15,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key=f"canvas_{st.session_state.canvas_key}",
    return_image_data=True
)

# Prediction
if canvas_result.image_data is not None:

    image = canvas_result.image_data
    image = Image.fromarray(image.astype("uint8")).convert("L")
    image = image.resize((28, 28))
    image = np.array(image)
    image = image.astype("float32") / 255.0
    image = image.reshape(1, 784)

    prediction = model.predict(image, verbose=0)

    predicted_digit = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    st.subheader(f"🎯 Prediction: {predicted_digit}")
    st.write(f"Confidence: **{confidence:.2f}%**")

    st.subheader("📊 Class Probabilities")

    for digit, probability in enumerate(prediction[0]):
        st.write(f"Digit {digit}: {probability * 100:.2f}%")
