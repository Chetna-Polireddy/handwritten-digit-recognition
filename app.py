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

# Drawing canvas
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=15,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
    return_image_data=True
)

# Prediction
if canvas_result.image_data is not None:

    # Convert canvas image to grayscale
    image = canvas_result.image_data
    image = Image.fromarray(image.astype("uint8")).convert("L")

    # Resize to MNIST size
    image = image.resize((28, 28))

    # Convert to NumPy array
    image = np.array(image)

    # Normalize pixels from 0-255 to 0-1
    image = image.astype("float32") / 255.0

    # Flatten 28x28 into 784 pixels
    image = image.reshape(1, 784)

    # Predict
    prediction = model.predict(image, verbose=0)

    # Get predicted digit
    predicted_digit = np.argmax(prediction)

    # Get confidence
    confidence = np.max(prediction) * 100

    # Display prediction
    st.subheader(f"Prediction: {predicted_digit}")
    st.write(f"Confidence: {confidence:.2f}%")

    # Display probabilities
    st.subheader("Class Probabilities")

    for digit, probability in enumerate(prediction[0]):
        st.write(f"Digit {digit}: {probability * 100:.2f}%")
