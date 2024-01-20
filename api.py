import uvicorn
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from load_model import pred_using_model, similarity_using_ae
from PIL import Image
import numpy as np
from torchvision import transforms

app = FastAPI()

# Define the image data model for segmentation
class Image(BaseModel):
    image: list

# Default endpoint
@app.get("/")
def index():
    return {"greeting": "Hello world"}

# Segment an image
@app.post("/segment")
def segment_image(image: Image):
    pred = pred_using_model(image.image, "models/model.pth")
    image.image = np.array(pred).tolist()
    return {"image": image.image}


# Computing similarity
@app.post("/similarity")
def compute_similarity(image: Image):
    # Convert the image to a tensor
    error = similarity_using_ae(image.image, 'models/autoencoder.pth')
    return {"similarity": error}