from fastapi import FastAPI, File, UploadFile, Form
from .utils import predict_accident_image, predict_severity, generate_summary, answer_question

app = FastAPI(title="Road Accident AI System")

@app.get("/")
def home():
    return {"message": "Welcome to Road Accident AI API"}

@app.post("/predict_image")
async def predict_image(file: UploadFile = File(...)):
    img_path = f"temp_{file.filename}"
    with open(img_path, "wb") as f:
        f.write(await file.read())
    result = predict_accident_image(img_path)
    return {"prediction": result}

@app.post("/predict_severity")
async def predict_text_severity(report: str = Form(...)):
    severity = predict_severity(report)
    return {"severity": severity}

@app.post("/summarize_report")
async def summarize_report(report: str = Form(...)):
    summary = generate_summary(report)
    return {"summary": summary}

@app.post("/qa")
async def qa(report: str = Form(...), question: str = Form(...)):
    answer = answer_question(question, report)
    return {"answer": answer}
