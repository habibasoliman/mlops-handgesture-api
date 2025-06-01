from fastapi import FastAPI, Request
from pydantic import BaseModel
from app.model import predictt
from prometheus_fastapi_instrumentator import Instrumentator
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter
import time
import os 
import pickle




app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Instrumentator().instrument(app).expose(app)

# Middleware to measure request processing time
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000  # in milliseconds
    response.headers["X-Process-Time-ms"] = str(process_time)
    return response

# Request body model
class LandmarkRequest(BaseModel):
    landmarks: list

# Counter for missing/incomplete landmark data
missing_landmarks_counter = Counter(
    "missing_landmarks_total", "Number of requests with missing landmarks"
)

# Prediction endpoint
@app.post("/predict")
async def predict_route(data: LandmarkRequest):
    # Validate landmark format
    for point in data.landmarks:
        if len(point) != 3:
            missing_landmarks_counter.inc()
            return {"error": "Invalid landmarks: each point must have 3 values."}
    try:
        prediction = predictt(data.landmarks)
        return {"gesture": prediction}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)
