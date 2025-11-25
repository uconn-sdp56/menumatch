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

Test at `POST http://127.0.0.1:8000/predict`

Deactivate the virtual environment when done
```bash
deactivate
```