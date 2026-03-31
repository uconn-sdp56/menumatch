### Running the API

Make sure you are inside the `/api` directory

Create a python virtual environment
```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies
```bash
pip install -r requirements.txt
```

Set environment variables
```bash
export OPENAI_API_KEY=your_key_here
```

Run the API
```bash
uvicorn app.main:app --reload
```

The API should now be running on port 8000.  
Use the `/predict` endpoint (i.e. `http://localhost:8000/predict`).  
Put inputs in binary request body, multipart form.  

| Key | Type | Example Value |
| :--- | :--- | :--- |
| image | File | choose an image file |
| dining_hall_id | Text | 16
| meal | Text | lunch |
| date | Text | 2026-03-31 |

Deactivate your virtual environment when done
```bash
deactivate
```
