from enum import Enum
from datetime import date
from typing import List

from fastapi import FastAPI, Request, Query
from pydantic import BaseModel

app = FastAPI()

class MealTime(str, Enum):
    BREAKFAST = "breakfast"
    LUNCH = "lunch"
    DINNER = "dinner"

class PortionEstimate(BaseModel):
    name: str
    id: str
    servings: float

@app.post("/predict", response_model=List[PortionEstimate])
async def predict(
    request: Request,
    dining_hall_id: str = Query(...),
    meal_time: MealTime = Query(...),
    meal_date: date = Query(...)
):
    # file_bytes = await request.body()
    return [
        PortionEstimate(name="Cross Trax French Fries", id="161069", servings=0.75),
        PortionEstimate(name="Fried Chicken Nuggets", id="111037", servings=2),
        PortionEstimate(name="Corn", id="171012", servings=1)
    ]

