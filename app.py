import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
from torchvision import transforms


# Load model

@st.cache_resource
def load_model():
    model = efficientnet_b4(weights=EfficientNet_B4_Weights.DEFAULT)
    model.classifier[0] = nn.Dropout(p=0.3)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)  # 2 output classes

    # Load the model weights from Reid's checkpoint
    checkpoint = torch.load("models/model_fold_1.pth", map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

model = load_model()

# Preprocessing

transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.56057299, 0.58049109, 0.74380801],
        std=[0.21038430, 0.19394357, 0.19913281]
    )
])

classes = ['benign', 'malignant']  # Replace with actual class names if different

# Streamlit UI

st.title("Skin Cancer Detection")
st.write("Upload a skin lesion image to get a prediction.")

uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1).numpy()[0]
        predicted_index = int(np.argmax(probabilities))
        predicted_label = classes[predicted_index]
        confidence = probabilities[predicted_index]

    st.markdown(f"### Prediction: `{predicted_label}`")
    st.markdown(f"**Confidence:** `{confidence:.2%}`")

