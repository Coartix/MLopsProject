import streamlit as st
from PIL import Image
import requests
import numpy as np

def send_request(image: np.ndarray):
    url = "http://127.0.0.1:8000/similarity"
    payload = {
        "image": image.tolist()
    }
    response = requests.post(url, json=payload)
    return response.json()

st.title('Image Similarity Check using Autoencoder')

# Upload the image
uploaded_image = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])
if uploaded_image is not None:
    # Convert the file to an image
    image = np.array(Image.open(uploaded_image))
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Send the image to the API
    if st.button('Check Similarity'):
        response = send_request(image)
        print(response)
        similarity = response['similarity']
        st.write(f'The similarity is {similarity:.2f}')
