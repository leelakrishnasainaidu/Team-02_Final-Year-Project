import os
import tempfile
import streamlit as st
import torch
import cv2
from PIL import Image
from utils.transforms import get_no_aug_transform
from models.generator import Generator
import torchvision.transforms.functional as TF
import time

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.info(f"💻 Using device: {'GPU' if torch.cuda.is_available() else 'CPU'}")

# Load Model
pretrained_path = "./checkpoints/trained_netG.pth"
netG = Generator().to(device)
netG.load_state_dict(torch.load(pretrained_path, map_location=device))
netG.eval()

# Inverse normalization for output image 
def inv_normalize(img_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)
    std = torch.tensor([0.229, 0.224, 0.225], device=device)
    return (img_tensor * std.view(1, 3, 1, 1) + mean.view(1, 3, 1, 1)).clamp(0, 1)

# Predict and process the image
def predict_image(img):
    start = time.time()
    transform = get_no_aug_transform()
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = inv_normalize(netG(img_tensor))
    end = time.time()
    st.success(f"✅ Processed in {end - start:.2f} seconds")
    return TF.to_pil_image(output.squeeze().cpu())

# Save uploaded file temporarily
def save_uploaded_file(uploaded_file):
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

# Streamlit App UI
st.title("🖼️ Cartoonizer - Image & Video Processing")

uploaded_file = st.file_uploader("Upload an Image or Video", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])

if uploaded_file:
    file_path = save_uploaded_file(uploaded_file)

    if uploaded_file.type.startswith("image"):
        image = Image.open(file_path).convert("RGB")
        st.image(image, caption="📤 Uploaded Image", use_container_width=True)

        if st.button("✨ Process Image"):
            with st.spinner("Processing... Please wait ⏳"):
                processed_image = predict_image(image)
                st.image(processed_image, caption="🎨 Processed Image", use_container_width=True)

                # Save cartooned image to buffer for download
                buffer = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                processed_image.save(buffer.name)
                buffer.flush()

                with open(buffer.name, "rb") as f:
                    st.download_button(
                        label="📥 Download Cartoonized Image",
                        data=f,
                        file_name="cartoonized_image.png",
                        mime="image/png"
                    )

    elif uploaded_file.type.startswith("video"):
        st.video(file_path)
        st.warning("⚠️ Video processing is currently not supported in this app.")
