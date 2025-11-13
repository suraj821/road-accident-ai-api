from fastapi import FastAPI, File, UploadFile
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import ProcessCollector, REGISTRY
from .utils import (
    predict_accident_image,
    predict_severity,
    generate_summary,
    answer_question,
    speech_to_text
)
import os

app = FastAPI(title="Road Accident AI System")

# ---------------- Prometheus Integration ----------------
# Register process-level metrics (CPU & Memory) in the Prometheus registry
REGISTRY.register(ProcessCollector(namespace="fastapi"))

# Initialize instrumentator for FastAPI metrics
instrumentator = Instrumentator(
    should_instrument_requests_inprogress=True,
    should_group_status_codes=False,
    should_group_untemplated=True
)

# Attach instrumentator and expose metrics at /metrics
instrumentator.instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)

# ---------------- Routes ----------------
@app.get("/")
def home():
    return {"message": "Welcome to Road Accident AI API"}

# ---------------- CV Endpoint ----------------
@app.post("/predict_image")
async def predict_image(file: UploadFile = File(...)):
    img_path = f"temp_{file.filename}"
    with open(img_path, "wb") as f:
        f.write(await file.read())
    result = predict_accident_image(img_path)
    os.remove(img_path)
    return {"prediction": result}

# ---------------- NLP Endpoints ----------------
@app.post("/predict_severity")
async def predict_text_severity(report: str = File(...)):
    severity = predict_severity(report)
    return {"severity": severity}

@app.post("/summarize_report")
async def summarize_report(report: str = File(...)):
    summary = generate_summary(report)
    return {"summary": summary}

@app.post("/qa")
async def qa(report: str = File(...), question: str = File(...)):
    answer = answer_question(question, report)
    return {"answer": answer}

# ---------------- Speech Recognition Endpoint ----------------
@app.post("/predict_severity_speech")
async def predict_severity_from_speech(file: UploadFile = File(...)):
    audio_path = f"temp_audio_{file.filename}"
    with open(audio_path, "wb") as f:
        f.write(await file.read())
    
    text = speech_to_text(audio_path)
    severity = predict_severity(text)
    
    os.remove(audio_path)
    return {"transcribed_text": text, "severity": severity}
