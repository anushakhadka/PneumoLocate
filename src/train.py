import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import PneumoniaDataset
from model import PneumoNet
from pathlib import Path


def main():
    # 1. Paths
    root = Path(__file__).resolve().parents[1]
    data_path = root / "data" / "processed"

    # 2. Data Loader
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    dataset = PneumoniaDataset(data_path, transform=transform)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    # 3. Model, Loss, Optimizer
    model = PneumoNet()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 4. Training Loop
    print("🚀 Starting AI training...")
    model.train()

    for images, labels in loader:
        optimizer.zero_grad()
        outputs = model(images).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print(f"Current Loss: {loss.item():.4f}")

    # 5. Save the Brain
    model_dir = root / "models"
    model_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), model_dir / "pneumo_model.pth")
    print(f"✅ Training complete! Brain saved to {model_dir}/pneumo_model.pth")


if __name__ == "__main__":
    main()
