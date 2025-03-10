import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import os
from model.unet import *
from model.unet_plus_plus import *
from model.backboned_unet import *

SAVE_PATH = {
    'unet': os.path.join('checkpoints', 'unet'),
    'unet_plus_plus': os.path.join('checkpoints', 'unet_plus_plus'),
    'backboned_unet': os.path.join('checkpoints', 'backboned_unet')
}

BACKBONE_OPTIONS = ["vgg16", "vgg19", "resnet18", "resnet34", "densenet121", "densenet169"]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@st.cache_resource
def load_model(model_name, backbone_name=None):
    """Load model based on user selection (cached for efficiency)."""
    if model_name == 'unet':
        model = Unet()
    elif model_name == 'unet_plus_plus':
        model = NestedUNet(num_classes=1)
    elif model_name == 'backboned_unet' and backbone_name:
        model = BackbonedUnet(backbone_name=backbone_name)
    else:
        return None  # Invalid model choice

    model.to(device)
    model_path = os.path.join(
        SAVE_PATH[model_name], 
        'best.pth' if model_name != 'backboned_unet' else f"{backbone_name}_best.pth"
    )

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    
    model.eval()  # Set model to evaluation mode
    return model

def preprocess_images(images):
    """Convert images to batch tensor format (B, C, H, W)."""
    processed_images = []
    
    for image in images:
        image = np.array(image)  # Convert PIL image to NumPy array

        # Ensure 3 channels (convert grayscale to RGB)
        if len(image.shape) == 2:  
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Convert HxWxC to CxHxW
        image = image.transpose(2, 0, 1)  # (H, W, C) → (C, H, W)

        # Normalize (scale pixel values to [0,1])
        image = image.astype(np.float32) / 255.0  

        processed_images.append(torch.from_numpy(image))

    return torch.stack(processed_images)  # Convert list to batch tensor

def batch_inference(model, images):
    """Perform segmentation for a batch of images (cached to avoid re-inference)."""
    with torch.no_grad():
        masks = model(images.to(device)).sigmoid().cpu().numpy()  # (B, 1, H, W)
    return masks.squeeze(1)  # Remove channel dim → (B, H, W)

def apply_threshold(masks, threshold):
    """Apply threshold to cached masks and return overlay images."""
    segmented_images = []
    
    for mask, original in zip(masks, st.session_state.original_images):
        binary_mask = (mask > threshold).astype(np.uint8) * 255
        mask_colored = cv2.applyColorMap(binary_mask, cv2.COLORMAP_JET)
        mask_colored = cv2.cvtColor(mask_colored, cv2.COLOR_BGR2RGB)

        # Blend with original image
        alpha = 0.5
        overlayed_image = cv2.addWeighted(original, 1 - alpha, mask_colored, alpha, 0)
        segmented_images.append(overlayed_image)

    return segmented_images

def main():
    st.title("MRI Segmentation App 🧠")

    # Sidebar - Model Selection
    st.sidebar.header("Model Selection")
    model_name = st.sidebar.selectbox("Choose a model:", ["unet", "unet_plus_plus", "backboned_unet"])
    backbone_name = None
    if model_name == "backboned_unet":
        backbone_name = st.sidebar.selectbox("Select a backbone:", BACKBONE_OPTIONS)

    # Load Model
    model = load_model(model_name, backbone_name)

    # Upload multiple MRI images
    uploaded_files = st.file_uploader(
        "Upload MRI images (JPG, PNG, JPEG)", 
        type=["jpg", "png", "jpeg"], 
        accept_multiple_files=True
    )

    if uploaded_files:
        # Convert images
        images = [Image.open(file).resize((256, 256)) for file in uploaded_files]
        processed_images = preprocess_images(images)

        # Check if files changed
        new_file_names = [file.name for file in uploaded_files]
        if (
            "masks" not in st.session_state 
            or st.session_state.model_name != model_name 
            or st.session_state.backbone_name != backbone_name
            or st.session_state.uploaded_file_names != new_file_names
        ):
            st.session_state.masks = batch_inference(model, processed_images)
            st.session_state.model_name = model_name
            st.session_state.backbone_name = backbone_name
            st.session_state.uploaded_file_names = new_file_names  # Store file names
            st.session_state.original_images = [np.array(img) for img in images]

        # Sidebar - Threshold Slider
        threshold = st.sidebar.slider("Segmentation Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

        # Apply threshold to cached masks
        segmented_images = apply_threshold(st.session_state.masks, threshold)

        # Display images
        for original, segmented in zip(st.session_state.original_images, segmented_images):
            col1, col2 = st.columns(2)
            with col1:
                st.image(original, caption="Original MRI Scan", use_container_width=True)
            with col2:
                st.image(segmented, caption="Segmented MRI", use_container_width=True)

if __name__ == "__main__":
    main()
