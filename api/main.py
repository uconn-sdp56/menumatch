from fastapi import FastAPI

app = FastAPI()

@app.post("/predict")
def predict():
    return {"message": "Hello from the endpoint"}