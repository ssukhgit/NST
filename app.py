import streamlit as st
from PIL import Image
from WCT.util import *
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image
from WCT.style_transfer import style_transfer
from io import BytesIO
import os
import torch
import sys
import subprocess

# Configuration for the WCT (whitening and coloring transform), pretrained encoders and decoders
config = {
    'fineSize': 512,
    'vgg1': 'WCT/models/vgg_normalised_conv1_1.pth',
    'vgg2': 'WCT/models/vgg_normalised_conv2_1.pth',
    'vgg3': 'WCT/models/vgg_normalised_conv3_1.pth',
    'vgg4': 'WCT/models/vgg_normalised_conv4_1.pth',
    'vgg5': 'WCT/models/vgg_normalised_conv5_1.pth',
    'decoder5': 'WCT/models/feature_invertor_conv5_1.pth',
    'decoder4': 'WCT/models/feature_invertor_conv4_1.pth',
    'decoder3': 'WCT/models/feature_invertor_conv3_1.pth',
    'decoder2': 'WCT/models/feature_invertor_conv2_1.pth',
    'decoder1': 'WCT/models/feature_invertor_conv1_1.pth',
}

# Title
st.title("Neural Style Transfer App")

text_block = """
This app allows you to perform neural style transfer using the Whitening and Coloring 
Transform (WCT) and CycleGAN algorithms. Press 'Transfer any style to any image' to 
transfer the style of one image to another, or 'Transform zebra 🦓 to horse 🐎' to 
convert a zebra image to a horse image using CycleGAN.
"""

st.write(text_block)

# Initialize session state for button tracking
if "button1_clicked" not in st.session_state:
    st.session_state.button1_clicked = False
if "button2_clicked" not in st.session_state:
    st.session_state.button2_clicked = False

# Choose type of transfer
col1, col2 = st.columns(2)

with col1:
    if st.button("Transfer any style to any image", use_container_width=True):
        st.session_state.button1_clicked = True
        st.session_state.button2_clicked = False  # Reset the other button state

with col2:
    if st.button("Transform zebra 🦓 to horse 🐎", use_container_width=True):
        st.session_state.button2_clicked = True
        st.session_state.button1_clicked = False  # Reset the other button state

# Handle Button 1 Actions
if st.session_state.button1_clicked:

    # Upload Content and Style Images
    content_image = st.file_uploader("Upload Content Image", type=["png", "jpg", "jpeg"])
    style_image = st.file_uploader("Upload Style Image", type=["png", "jpg", "jpeg"])

    if content_image and style_image:
        content_image = Image.open(content_image).convert("RGB")
        style_image = Image.open(style_image).convert("RGB")

        # Display the uploaded images
        st.image(content_image, caption="Content Image", use_container_width=True)
        st.image(style_image, caption="Style Image", use_container_width=True)

        # Preprocessing pipeline
        preprocess = transforms.Compose([
            transforms.Resize(512),  # Adjust size dynamically if needed
            transforms.ToTensor(),
        ])

        # Preprocess Images
        contentImg = preprocess(content_image).unsqueeze(0)  # Add batch dimension
        styleImg = preprocess(style_image).unsqueeze(0)

        # Parameters
        min_value = 0.0
        max_value = 1.0

        # Style Weight Slider
        style_weight = st.slider(
            "Content/Style Weight",
            min_value=min_value,
            max_value=max_value,
            value=0.5,
            step=0.01,
        )

        # Add custom labels with columns
        col1, col2, col3 = st.columns([2, 6, 2])
        col1.markdown("<p style='font-weight: bold;'>More content</p>", unsafe_allow_html=True)
        col3.markdown("<p style='text-align: right; font-weight: bold;'>More style</p>", unsafe_allow_html=True)

        # Process Button
        if st.button("Generate Styled Image"):
            with st.spinner("Processing..."):
               
                # Initialize WCT model with configuration to pass to style_transfer function
                wct = WCT(config)

                # Perform Style Transfer
                with torch.no_grad():
                    output_image = style_transfer(contentImg, styleImg, style_weight, wct)
                st.success("Style Transfer Complete!")

                output_image = torch.clamp(output_image, 0, 1)

                # Display the result
                output_image = to_pil_image(output_image.squeeze(0))
                st.image(
                    output_image,
                    caption="Styled Image",
                    use_container_width=True
                )
               
                # Save button
                buf = BytesIO()
                output_image.save(buf, format="JPEG")
                byte_im = buf.getvalue()

                btn = st.download_button(
                    label="Download image",
                    data=byte_im,
                    file_name="styled_image.jpg",
                    mime="image/jpeg"
                )

# Handle Button 2 Actions
if st.session_state.button2_clicked:

    uploaded_file = st.file_uploader("Upload a zebra image", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        # Show the user-uploaded image
        st.image(
            uploaded_file,
            caption="Uploaded Zebra Image",
            use_container_width=True
        )

        if st.button("Convert to Horse"):
            with st.spinner("Processing..."):
                # 1. Save the uploaded image to CycleGAN/input/ so CycleGAN can read it
                input_image_path = "input/horse2zebra/testB/test.jpg"

                # Ensure the directory exists
                output_dir = os.path.dirname(input_image_path)
                os.makedirs(output_dir, exist_ok=True)

                with open(input_image_path, "wb") as f:
                    f.write(uploaded_file.getvalue())

                # 2. Call the CycleGAN test script.
                #    Adjust the paths/args to match your model name and checkpoint directory.
                python_path = sys.executable  # Ensure same Python environment

                command = [
                    python_path,
                    "CycleGAN/test.py",
                    "--dataroot", "./input/horse2zebra/testB",
                    "--checkpoints_dir", "./checkpoints",
                    "--name", "h2z_2_cyclegan",
                    "--model", "test",
                    "--no_dropout",
                    "--preprocess", "none",
                    "--results_dir", "./results",
                    "--gpu_ids", "-1"
                ]
                subprocess.run(command)

                # 3. The output image appears in:
                #    CycleGAN/results/h2z_2_cyclegan/test_latest/images/test_fake_B.png
                result_image_path = "results/h2z_2_cyclegan/test_latest/images/test_fake.png"
                st.success("Here is the horse!")

                if os.path.exists(result_image_path):
                    # 4. Load and display the generated horse image
                    result_image = Image.open(result_image_path)
                    st.image(
                        result_image,
                        caption="Converted Horse Image",
                        use_container_width=True
                    )
                else:
                    st.error("CycleGAN did not produce an output image. Check the console for errors.")
