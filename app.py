import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import os
from PIL import Image
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
from torchvision import transforms

MODEL_PATH = "models/model_fold_1.pth"
CLASS_NAMES = ['benign', 'malignant']
IMAGE_SIZE = (384, 384)  # Resize input to match EfficientNet-B4 input size
NORMALIZE_MEAN = [0.5620089173316956, 0.5811969041824341, 0.7454625368118286]
NORMALIZE_STD = [0.21154886484146118, 0.19519373774528503, 0.20036040246486664]

#Preprocessing

transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
])

#Model portion

@st.cache_resource
def load_model(path=MODEL_PATH):
    try:
        model = efficientnet_b4(weights=EfficientNet_B4_Weights.DEFAULT)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(model.classifier[1].in_features, len(CLASS_NAMES))
        )
        checkpoint = torch.load(path, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model
    except FileNotFoundError:
        st.error(f"Model file not found at: `{path}`.")
        raise
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        raise

#Prediction portion

def predict(image, model):
    try:
        input_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1).numpy()[0]
            predicted_index = int(np.argmax(probabilities))
            return CLASS_NAMES[predicted_index], probabilities[predicted_index]
    except Exception as e:
        st.error("An error occurred during prediction.")
        st.exception(e)
        return None, None

#Streamlit portion

st.set_page_config(page_title="Skin Cancer Detector", layout="centered")
st.title("🩺 Skin Cancer Detection")
st.write("Upload a skin lesion image and receive a model prediction.")

uploaded_file = st.file_uploader("Choose a lesion image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        model = load_model()
        label, confidence = predict(image, model)

        if label:
            st.success(f"**Prediction:** `{label}`")
            st.info(f"**Confidence:** `{confidence:.2%}`")
    except Exception as e:
        st.error("Something went wrong with processing this image.")
        st.exception(e)
