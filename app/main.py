from fastapi import FastAPI, File, UploadFile
from .utils import (
    predict_accident_image, 
    predict_severity, 
    generate_summary, 
    answer_question,
    speech_to_text
)
import shutil
import os
from pathlib import Path

app = FastAPI(title="Road Accident AI System")

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
    # dir = Path("D:\\Suraj\\MTECH Learning\\Semester 2\\API Driven\\Assignments\\Assignment2\\road-accident-ai-api")
    audio_path = f"temp_audio_{file.filename}"
    with open(audio_path, "wb") as f:
        f.write(await file.read())
    
    # Convert speech to text
    text = speech_to_text(audio_path)
    
    # Run existing NLP prediction
    severity = predict_severity(text)
    
    os.remove(audio_path)
    return {"transcribed_text": text, "severity": severity}
