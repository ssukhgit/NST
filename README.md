# Neural Style Transfer with WCT & CycleGAN 
This project combines state-of-the-art neural style transfer techniques using both [PytorchWCT](https://github.com/sunshineatnoon/PytorchWCT)  and [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)  to produce impressive style transfer results. An interactive demo of the project is deployed on [Streamlit](https://styletransferwctcyclegan.streamlit.app/) .
## Overview 

This project integrates two powerful approaches:
 
- **CycleGAN-based Style Transfer:**  Adapted from [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) , enabling image-to-image translation (e.g., zebra to horse and vice versa).
 
- **WCT-based Style Transfer:**  Adapted from [PytorchWCT](https://github.com/sunshineatnoon/PytorchWCT) , providing a Whitening and Coloring Transform that enable transfer of any style to any image.

## Features 
 
- **Interactive Web Interface:** 
The app is built using Streamlit, allowing users to easily upload images and view stylized results.
 
- **Multiple Style Transfer Methods:** 
Combines CycleGAN and WCT methodologies for versatile style transfer.
 
- **Real-Time Processing:** 
The app processes images on the fly and provides immediate visual feedback.

- **Custom training:**
CycleGAN has been trained, the resulting generator is stored in /checkpoints directory

## Deployment 

This project is deployed on Streamlit Cloud. Every time the repository is updated, the app is redeployed automatically. Files saved during runtime (e.g., user-uploaded images) are stored in an ephemeral container, so they will not persist across sessions.
Visit the live app here: [Streamlit App](https://styletransferwctcyclegan.streamlit.app/) .

## Credits 
 
- **pytorch-CycleGAN-and-pix2pix:** 
Original repository by [junyanz](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) .
 
- **PytorchWCT:** 
Original repository by [sunshineatnoon](https://github.com/sunshineatnoon/PytorchWCT) .

Special thanks to the respective authors and contributors for making these projects available.
