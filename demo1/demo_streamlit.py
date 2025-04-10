import streamlit as st
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

def generate_dummy_attention(image):
    img = np.array(image)
    heatmap = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

    center_x, center_y = img.shape[1] // 2, img.shape[0] // 2
    cv2.circle(heatmap, (center_x, center_y), 60, 255, -1)

    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)

    return overlay

st.title("🧠 Language-Guided Visual Attention Demo (Simulated)")

prompt = st.text_input("Enter your task prompt (e.g., 'Find the object used for writing')")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file and prompt:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original Image", use_container_width=True)

    st.markdown("### 🧭 Prompt:")
    st.markdown(f"> **{prompt}**")

    st.markdown("### 🔥 Simulated Attention Map:")
    simulated_attention = generate_dummy_attention(image)
    st.image(simulated_attention, use_container_width=True)
