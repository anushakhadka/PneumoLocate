import gradio as gr
import torch
from PIL import Image
from torchvision import transforms
from model import PneumoNet
from pathlib import Path

# 1. Load the trained brain
root = Path(__file__).resolve().parents[1]
model = PneumoNet()
model.load_state_dict(
    torch.load(root / "models" / "pneumo_model.pth", map_location="cpu")
)
model.eval()


# 2. Define the 'Inference' function
def predict_pneumonia(img):
    # Prepare the image
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        prob = model(img_tensor).item()

    # Return a dictionary for the 'Label' component
    return {"Pneumonia": prob, "Healthy": 1 - prob}


# 3. Build the Interface
demo = gr.Interface(
    fn=predict_pneumonia,
    inputs=gr.Image(type="pil", label="Upload Chest X-Ray"),
    outputs=gr.Label(num_top_classes=2, label="AI Diagnosis"),
    title="PneumoLocate: AI-Powered Lung Analysis",
    description="Upload a chest X-ray image to check for signs of Pneumonia. Disclaimer: For educational use only.",
    theme="soft",
)

if __name__ == "__main__":
    # We add a print statement so you know for sure it's starting
    print("Connecting to Gradio server... please wait.")
    demo.launch(share=True, show_error=True)
