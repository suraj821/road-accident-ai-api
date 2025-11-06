import torch
from torchvision import models, transforms
from PIL import Image
import os

# ----------------------------
# 1. Model Setup
# ----------------------------
# Same transform used during training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Classes should match your dataset folder names
class_names = ['Accident', 'No_Accident']

# Load ResNet50 model structure
model = models.resnet50(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 2)

# Load fine-tuned weights
MODEL_PATH = os.path.join(os.path.dirname(__file__), "resnet50_accident.pt")
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
model.eval()

# ----------------------------
# 2. Prediction Function
# ----------------------------
def predict_accident_image(image_path: str) -> str:
    """Predict whether the image shows an Accident or No_Accident."""
    try:
        image = Image.open(image_path).convert('RGB')
        img_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(img_tensor)
            _, preds = torch.max(outputs, 1)

        predicted_class = class_names[preds.item()]
        return predicted_class
    except Exception as e:
        return f"Error processing image: {e}"


# ----------------------------
# 3. Dummy Stubs (keep your existing ones)
# ----------------------------
def predict_severity(report: str) -> str:
    # Example rule-based or model-based logic
    if "major" in report.lower() or "serious" in report.lower():
        return "High"
    elif "minor" in report.lower():
        return "Low"
    else:
        return "Medium"


def generate_summary(report: str) -> str:
    # Simple text summarization stub
    return " ".join(report.split()[:15]) + "..."


def answer_question(question: str, context: str) -> str:
    # Simple QA stub for demo
    if "where" in question.lower():
        for word in context.split():
            if word.istitle():
                return word
        return "Location not found."
    return "Answer not available."
