import torch
from torchvision import models, transforms
from PIL import Image
import whisper
import os


ffmpeg_path = "/usr/local/bin/ffmpeg"  
os.environ['PATH'] += f':{os.path.dirname(ffmpeg_path)}'


# ----------------------------
# Computer Vision Model (unchanged)
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
class_names = ['Accident', 'No_Accident']
model = models.resnet50(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 2)
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
# Speech Recognition
# ----------------------------
# Load Whisper model once
whisper_model = whisper.load_model("large")  # or "small", "medium", "large"

def speech_to_text(audio_file_path: str) -> str:
    """
    Convert audio file (.wav, .mp3) to text using Whisper.
    """
    result = whisper_model.transcribe(audio_file_path)
    return result['text']
