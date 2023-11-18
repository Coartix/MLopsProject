import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

app = FastAPI()

class Image(BaseModel):
    image: list

@app.post("/segment")
async def segment_image(image: Image):
    # Simulate some asynchronous processing (replace with actual logic)
    import asyncio
    await asyncio.sleep(2)

    # return the image list to test the API
    return {"image": image.image}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)