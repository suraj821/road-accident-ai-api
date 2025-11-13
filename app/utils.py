import torch
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from PIL import Image
import whisper
import os

# ----------------------------
# FFmpeg path for Whisper (if needed)
# ----------------------------
ffmpeg_path = "/usr/local/bin/ffmpeg"
os.environ['PATH'] += f':{os.path.dirname(ffmpeg_path)}'

# ----------------------------
# Computer Vision Model
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

class_names = ['Accident', 'No_Accident']

# Load ResNet50 without deprecated 'pretrained' argument
weights = None  # or ResNet50_Weights.DEFAULT if you want pretrained weights
model = models.resnet50(weights=weights)

# Update the final layer for 2 classes
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 2)

# Load your trained model weights
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models/resnet50_accident.pt")
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
model.eval()

def predict_accident_image(image_path: str) -> str:
    image = Image.open(image_path).convert('RGB')
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        _, preds = torch.max(outputs, 1)
    return class_names[preds.item()]

# ----------------------------
# NLP stubs
# ----------------------------
def predict_severity(text: str) -> str:
    if "major" in text.lower() or "serious" in text.lower():
        return "High"
    elif "minor" in text.lower():
        return "Low"
    else:
        return "Medium"

def generate_summary(text: str) -> str:
    return " ".join(text.split()[:15]) + "..."

def answer_question(question: str, context: str) -> str:
    if "where" in question.lower():
        for word in context.split():
            if word.istitle():
                return word
        return "Location not found."
    return "Answer not available."

# ----------------------------
# Speech Recognition with Whisper
# ----------------------------
whisper_model = whisper.load_model("large")  # choose "small", "medium", "large"

def speech_to_text(audio_file_path: str) -> str:
    result = whisper_model.transcribe(audio_file_path)
    return result['text']
