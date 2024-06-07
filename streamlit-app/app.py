import streamlit as st
import time
from PIL import Image
import os

# Set up the app title and description
st.title("Skin Lesion Detection with Deep Learning")

# Additional information about the project
st.write("""
## About Skin Lesion Detection
Skin lesion detection is a crucial task in dermatology for diagnosing skin cancer and other skin conditions.
Deep learning models can assist dermatologists by providing accurate and fast segmentation of skin lesions from
images.
In this application, we demonstrate how a deep learning model can segment skin lesions from sample images.
""")

st.write("""
This application demonstrates a deep learning model for skin lesion instance segmentation.
You can select a sample image to see the model's prediction and compare it with the ground truth segmentation
mask.
""")

# List of sample images
sample_images = ["sample1.jpg", "sample2.jpg", "sample3.jpg", "sample4.jpg", "sample5.jpg"]

# Input widget to select an image
selected_image = st.selectbox("Choose a sample image:", sample_images)

# Simulate model loading
if st.button("Load Model and Predict"):
    with st.spinner('Loading model...'):
        time.sleep(2)  # Simulate loading time

    # Load images
    sample_image_path = f"./data/sample/{selected_image}"
    prediction_image_path = f"./data/prediction/{selected_image}"

    # Adjust the file extension for the ground truth images
    ground_truth_image_name = os.path.splitext(selected_image)[0] + ".png"
    ground_truth_image_path = f"./data/ground_truth/{ground_truth_image_name}"

    sample_image = Image.open(sample_image_path)
    prediction_image = Image.open(prediction_image_path)
    ground_truth_image = Image.open(ground_truth_image_path)

    # Display images
    st.write("### Original Sample Image")
    st.image(sample_image, caption="Sample Image", use_column_width=True)

    st.write("### Prediction Segmentation Mask")
    st.image(prediction_image, caption="Prediction Mask", use_column_width=True)

    st.write("### Ground Truth Segmentation Mask")
    st.image(ground_truth_image, caption="Ground Truth Mask", use_column_width=True)

