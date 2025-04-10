import streamlit as st

st.title("Language-Guided Visual Attention Demo")

uploaded_file = st.file_uploader("Upload an image")
prompt = st.text_input("Enter your task prompt (e.g., 'Find the knife')")

if uploaded_file and prompt:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Input Image", use_column_width=True)


