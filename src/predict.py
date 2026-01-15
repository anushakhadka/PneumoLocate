import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from model import PneumoNet  # Your AI brain
from pathlib import Path


def predict_and_show(image_path):
    # 1. Load the trained brain
    model = PneumoNet()
    # We use 'map_location' in case you don't have a GPU
    model.load_state_dict(
        torch.load("models/pneumo_model.pth", map_location=torch.device("cpu"))
    )
    model.eval()  # Tell the brain it's in "Review Mode"

    # 2. Prepare the image (the same way we did for training)
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)  # Add a 'batch' dimension

    # 3. Make the prediction
    with torch.no_grad():
        probability = model(img_tensor).item()

    # 4. SHOW THE VISUAL
    plt.imshow(img)
    result_text = f"Pneumonia Probability: {probability:.2%}"
    color = "red" if probability > 0.5 else "green"
    plt.title(result_text, color=color, fontsize=16)
    plt.axis("off")
    print(f"Result: {result_text}")
    plt.show()


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    # Point this to one of your preview images
    sample_image = root / "data" / "processed" / "dicom_preview.png"

    if sample_image.exists():
        predict_and_show(sample_image)
    else:
        print("Image not found! Run your dicom_to_png script first.")
