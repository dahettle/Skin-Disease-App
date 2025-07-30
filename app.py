import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import models, transforms
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights

correct_size = (380,380) ## Will be used to process the uploaded image to the correct size to work with efficientnet_b4
computed_mean = [0.5620089173316956, 0.5811969041824341, 0.7454625368118286] ## Values taken directly from our data_normalization.py
computed_std = [0.21154886484146118, 0.19519373774528503, 0.20036040246486664] ## Values taken directly from our data_normalization.py
saved_path = "models/model_fold_1.pth" ## Saved location of the required trained .pth file
disease_classification = ['benign', 'malignant'] ## Skin disease classifications to be used

## Function call to process image the user uploads ##
processed_image = transforms.Compose([
    transforms.Resize(correct_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=computed_mean, std=computed_std),
])

## Portion of code that takes the user loaded image and sends it through the trained model ##
@st.cache_resource
def call_model(path=saved_path):
    try:
        trained_model=efficientnet_b4(weights=EfficientNet_B4_Weights.DEFAULT)
          # freeze other layers
        for param in trained_model.parameters():
            param.requires_grad = False
      
        trained_model.classifier[0] = nn.Dropout(p=0.3)
        trained_model.classifier[1] = nn.Linear(trained_model.classifier[1].in_features, len(disease_classification))
    
        checkpoint = torch.load(path, map_location = torch.device("cpu"))
        trained_model.load_state_dict(checkpoint['model_state_dict'])
        #epoch = checkpoint['epoch']
        #loss = checkpoint['loss']
        trained_model.eval()
        return trained_model
    except Exception as e:
        st.error(f"The user interface has experienced and error, please try again")

## Portion of code responsible for prediction and evaluation
def evaluation(loaded_image,trained_model):
    try:
        input = processed_image(loaded_image)
        input = input.unsqueeze(0)
        with torch.no_grad():
            output = trained_model(input)
            class_prob = torch.softmax(output, dim=1)[0]
            # Used to classify the most probable class and provide its level of accuracy
            topprob, topclass = torch.max(class_prob, dim=0)
            # Returns class name and accuracy
            return disease_classification[topclass.item()], topprob.item()

    except Exception as e:
        st.error(f"Evaluation process failed")

## Streamlit portion of the code ##
st.title("Skin Disease Prediction Model") # Sets the title of the UI
st.write("Please upload an image to begin") # Guides the user what to do next

loaded_image = st.file_uploader("Image type must be _JPEG_ or _PNG_") # Prompt seen on webpage to instruct user on accepted image formats
if loaded_image is not None: # Acceptable image has been uploaded
    display_image = Image.open(loaded_image)
    display_image.convert('RGB')

    trained_model = call_model()
    classification, probability = evaluation(display_image, trained_model)
    st.info(f"Classified as: '{classification}'")
    st.info(f"Level of accuracy: '{probability:.2%}'")
