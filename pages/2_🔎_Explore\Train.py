##Load libraries
import streamlit as st
from PIL import Image


im_example = Image.open("images/output.png").resize((900, 400))

st.title("Here is the dataset we used to train our model:")
st.image(im_example)

st.title("Here is the model we used:")
st.image("images/deeplab.jpg")