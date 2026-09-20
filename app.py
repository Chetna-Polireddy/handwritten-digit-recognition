import streamlit as st
import tensorflow as tf

st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="🔢"
)

st.title("🔢 Handwritten Digit Recognition")

st.write("Loading the trained neural network...")

model = tf.keras.models.load_model("handwritten_digit_model.keras")

st.success("✅ Neural network loaded successfully!")
st.write("Model is ready for digit prediction.")
