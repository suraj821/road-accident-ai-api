import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline, T5ForConditionalGeneration, T5Tokenizer
from torchvision import models, transforms
from PIL import Image

# --- Image Classification Model ---
resnet_model = models.resnet50(pretrained=True)
resnet_model.eval()

# --- Accident Severity Prediction (Text) ---
tokenizer_bert = AutoTokenizer.from_pretrained("bert-base-uncased")
bert_model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)  # Low, Medium, High

# --- Text Summarization Model ---
t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")
t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")

# --- Question Answering Pipeline ---
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# --- Preprocessing for Image ---
def preprocess_image(img_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    image = Image.open(img_path).convert("RGB")
    return transform(image).unsqueeze(0)
