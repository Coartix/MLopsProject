#Load libraries needed
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from streamlit_lottie import st_lottie
from PIL import Image


# Load the model
# model = joblib.load('models/model.pkl')

#define app section
header=st.container()
prediction=st.container()

# Define the Lottie animation URL
lottie_animation_url = "https://lottie.host/a9148890-ea55-4736-a4b7-717b394c44ad/Kh2Ihic5Q6.json"

#define header
with header:
    header.title("Semantic Segmentation")

    st_lottie(lottie_animation_url,height=200)
    header.write("On this page, you can segment your images")


# Create lists
inputs = ["date", "holiday", "locale", "transferred", "onpromotion"]
categorical = ["holiday", "locale", "transferred"]


# Set up prediction container
with st.expander("Make a segmentation", expanded=True):
    # Create input fields of an image
    image = st.file_uploader(label="Upload an image", type=['png', 'jpg', 'jpeg'])
    # Get image info an np array
    if image is not None:
        image = np.array(Image.open(image))
        print(image.shape)

# Use st.dataframe to display a dataframe
