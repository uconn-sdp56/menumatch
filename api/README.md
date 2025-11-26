## How to run the API

First, make sure you are inside the /api directory

Activate a python virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

Run the API

```bash
uvicorn main:app --reload
```

The API should now be running on port 8000.

Example request (put image in binary request body): `POST http://localhost:8000/predict?dining_hall_id=5&meal_time=lunch&meal_date=2025-11-25`

Deactivate the virtual environment when done
```bash
deactivate
```