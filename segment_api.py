import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from load_model import pred_using_model
import numpy as np

app = FastAPI()

class Image(BaseModel):
    image: list

@app.post("/segment")
def segment_image(image: Image):
    # Create thread to run the model
    pred = pred_using_model(image.image, "models/model.pth")
    image.image = np.array(pred).tolist()

    return {"image": image.image}
