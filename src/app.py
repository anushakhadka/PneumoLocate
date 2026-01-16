import gradio as gr
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from model import PneumoNet
from pathlib import Path

# 1. Setup Model and Paths
root = Path(__file__).resolve().parents[1]
model = PneumoNet()

# Load the trained brain (model weights)
model_path = root / "models" / "pneumo_model.pth"
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
model.eval()

# 2. Setup Grad-CAM (Targeting the final convolutional layer)
# This "Safe Search" finds your ResNet layers automatically to avoid AttributeErrors
try:
    internal_name = next(name for name, child in model.named_children())
    internal_model = getattr(model, internal_name)
    target_layers = [internal_model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
except Exception as e:
    print(f"Grad-CAM Setup Error: {e}")
    cam = None


def predict_and_visualize(img):
    # Image Preprocessing
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )
    norm_transform = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    img_tensor = transform(img).unsqueeze(0)
    input_tensor = norm_transform(img_tensor)

    # AI Prediction
    with torch.no_grad():
        prob = model(input_tensor).item()

    # Diagnosis Labels
    labels = {"Pneumonia": prob, "Healthy": 1 - prob}

    # Generate Heatmap if Grad-CAM is active
    if cam is not None:
        targets = [ClassifierOutputTarget(0)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]

        # Overlay heatmap on resized original image
        img_array = np.array(img.resize((224, 224))).astype(np.float32) / 255
        visualization = show_cam_on_image(img_array, grayscale_cam, use_rgb=True)
        return labels, visualization
    else:
        return labels, img.resize((224, 224))


# 3. Professional Web Interface
demo = gr.Interface(
    fn=predict_and_visualize,
    inputs=gr.Image(type="pil", label="Upload Chest X-Ray"),
    outputs=[
        gr.Label(num_top_classes=2, label="AI Confidence Score"),
        gr.Image(label="Diagnostic Heatmap (Infection Focus)"),
    ],
    title="PneumoLocate Pro: Medical Vision AI",
    description="Upload a chest X-ray to analyze for signs of pneumonia. Red areas on the heatmap indicate where the AI sees potential consolidation.",
    theme="glass",  # Using Gradio 6.0 compatible theme setup
)

if __name__ == "__main__":
    print("🚀 Launching PneumoLocate Pro...")
    demo.launch(share=True)
