from io import BytesIO

from fastapi import FastAPI, File, Form, UploadFile
from PIL import Image

from .pipeline import predict

app = FastAPI()


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/predict")
async def predict_endpoint(
    image: UploadFile = File(...),
    dining_hall_id: str = Form(...),
    meal: str = Form(...),
    date: str = Form(...),
):
    raw = await image.read()
    pil_image = Image.open(BytesIO(raw)).convert("RGB")
    servings = predict(pil_image, dining_hall_id, meal, date)
    return {"servings": servings}