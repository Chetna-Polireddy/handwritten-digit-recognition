import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# Load trained model
model = tf.keras.models.load_model(
    "handwritten_digit_model.keras",
    compile=False
)

# Page configuration
st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="🔢",
    layout="centered"
)

# Title
st.title("🔢 Handwritten Digit Recognition")
st.write("Draw a digit from 0 to 9 and let the neural network predict it!")

# Clear canvas
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

    # Convert drawing to grayscale
    image = canvas_result.image_data
    image = Image.fromarray(
        image.astype("uint8")
    ).convert("L")

    # Resize to MNIST size
    image = image.resize((28, 28))

    # Convert to NumPy array
    image = np.array(image)

    # Normalize pixels from 0-255 to 0-1
    image = image.astype("float32") / 255.0

    # Flatten 28x28 into 784 inputs
    image = image.reshape(1, 784)

    # Get model prediction
    prediction = model.predict(image, verbose=0)

    # Predicted digit
    predicted_digit = np.argmax(prediction[0])

    # Confidence
    confidence = np.max(prediction[0]) * 100

    # Display prediction
    st.subheader(f"🎯 Prediction: {predicted_digit}")
    st.write(f"Confidence: **{confidence:.2f}%**")

    # Class probabilities
    st.subheader("📊 Class Probabilities")

    for digit, probability in enumerate(prediction[0]):
        percentage = float(probability)

        st.write(
            f"**Digit {digit}: {percentage * 100:.2f}%**"
        )

        st.progress(percentage)
