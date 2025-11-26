from fastapi import FastAPI, Request, Query
from enum import Enum
from datetime import date

app = FastAPI()

class MealTime(str, Enum):
    breakfast = "breakfast"
    lunch = "lunch"
    dinner = "dinner"

@app.post("/predict")
async def predict(
    request: Request,
    dining_hall_id: str = Query(...),
    meal_time: MealTime = Query(...),
    meal_date: date = Query(...)
):
    file_bytes = await request.body()
    return {
        "message": "Hello from the endpoint",
        "meal_time": meal_time,
        "dining_hall_id": dining_hall_id,
        "date": meal_date,
        "file_size": len(file_bytes)
    }
