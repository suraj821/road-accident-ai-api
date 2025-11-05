import torch
from .models import resnet_model, preprocess_image, tokenizer_bert, bert_model, t5_model, t5_tokenizer, qa_pipeline

# Image classification
def predict_accident_image(img_path):
    img_tensor = preprocess_image(img_path)
    outputs = resnet_model(img_tensor)
    _, pred = torch.max(outputs, 1)
    return "Accident" if pred.item() == 1 else "No Accident"

# Text severity
def predict_severity(text):
    inputs = tokenizer_bert(text, return_tensors="pt", truncation=True, padding=True)
    outputs = bert_model(**inputs)
    pred = torch.argmax(outputs.logits, dim=1).item()
    return ["Low", "Medium", "High"][pred]

# Summarization
def generate_summary(text):
    inputs = t5_tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = t5_model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4)
    return t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Question Answering
def answer_question(question, context):
    return qa_pipeline(question=question, context=context)['answer']
