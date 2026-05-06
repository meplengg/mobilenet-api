from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from PIL import Image, UnidentifiedImageError
import io
import asyncio
from concurrent.futures import ProcessPoolExecutor

from app.model import Model
from app.preprocess import preprocess

app = FastAPI()

model = Model()
executor = ProcessPoolExecutor()

MAX_SIZE = 5 * 1024 * 1024

class PredictionResponse(BaseModel):
    class_id: int

def run_model(x):
    return model.predict(x)

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):

    if file.content_type not in [
        "image/jpeg",
        "image/png"
    ]:
        raise HTTPException(400,"Invalid file type")

    data = await file.read()

    if len(data) > MAX_SIZE:
        raise HTTPException(400,"File too large")

    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(400,"Corrupted image")

    x = preprocess(img)

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        executor,
        run_model,
        x
    )

    return {"class_id": result}