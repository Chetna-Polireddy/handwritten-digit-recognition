import streamlit as st
from streamlit_drawable_canvas import st_canvas

st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="🔢"
)

st.title("🔢 Handwritten Digit Recognition")

st.write("Draw a digit from 0 to 9 below:")

canvas_result = st_canvas(
    fill_color="black",
    stroke_width=15,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

st.write("Canvas loaded successfully! 🎉")
