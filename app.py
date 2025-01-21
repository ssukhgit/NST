import streamlit as st
from PIL import Image
import numpy as np

# Title
st.title("Neural Style Transfer App")

# Upload Content and Style Images
content_image = st.file_uploader("Upload Content Image", type=["png", "jpg", "jpeg"])
style_image = st.file_uploader("Upload Style Image", type=["png", "jpg", "jpeg"])

if content_image and style_image:
    content_image = Image.open(content_image).convert("RGB")
    style_image = Image.open(style_image).convert("RGB")
    
    # Display the uploaded images
    st.image(content_image, caption="Content Image", use_column_width=True)
    st.image(style_image, caption="Style Image", use_column_width=True)