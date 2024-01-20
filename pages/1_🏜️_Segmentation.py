import streamlit as st
import pandas as pd
import numpy as np
import joblib
from streamlit_lottie import st_lottie
from PIL import Image
import requests

def send_request(image: np.ndarray):
    url = "http://127.0.0.1:8000/segment"
    payload = {
        "image": image.tolist()
    }
    response = requests.post(url, json=payload)
    return response.json()

#### Streamlit section #####

header=st.container()
prediction=st.container()
lottie_animation_url = "https://lottie.host/a9148890-ea55-4736-a4b7-717b394c44ad/Kh2Ihic5Q6.json"
with header:
    header.title("Semantic Segmentation")
    st_lottie(lottie_animation_url,height=200)
    header.write("On this page, you can segment your images")


###### Set up prediction container ######

with st.expander("Make a segmentation", expanded=True):
    # Create input fields of an image
    image = st.file_uploader(label="Upload an image", type=['png', 'jpg', 'jpeg'])
    # Get image info an np array
    if image is not None:
        image = np.array(Image.open(image))
    if st.button("Segment the image") and image is not None:
        # Send the request and get the prediction
        response = send_request(image)
        image = response['image']
        st.write("The prediction is:")
        # Turn list into image
        l = response['image']
        l = np.array(l)
        # Display the image
        st.image(l)
