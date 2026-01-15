import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from torchvision import transforms


class PneumoniaDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        # 1. Path to your PNG images
        self.img_dir = Path(img_dir)
        self.img_paths = list(self.img_dir.glob("*.png"))
        self.transform = transform

        # 2. Extract labels from filenames (Industry trick!)
        # We assume if the filename contains 'pneumonia', it's Class 1 (Sick)
        # Otherwise, it's Class 0 (Healthy)
        self.labels = [
            1 if "pneumonia" in p.name.lower() else 0 for p in self.img_paths
        ]

    def __len__(self):
        # Returns the total number of images found
        return len(self.img_paths)

    def __getitem__(self, idx):
        # 3. Grab one image and its label
        img_path = self.img_paths[idx]
        label = self.labels[idx]

        # Open as RGB because ResNet expects 3 color channels
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)


if __name__ == "__main__":
    # Test the dataset
    root = Path(__file__).resolve().parents[1]
    data_path = root / "data" / "processed"

    # Simple transform: Resize and turn into math (Tensor)
    test_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    ds = PneumoniaDataset(data_path, transform=test_transform)
    print(f"✅ Found {len(ds)} images in processed folder.")

    if len(ds) > 0:
        img, lbl = ds[0]
        print(f"Sample Image Shape: {img.shape}")
        print(f"Sample Label: {lbl}")
