from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

def main():
    # 1. Setup paths relative to this file
    # .parent.parent moves from src/ up to the root folder PneumoLocate
    project_root = Path(__file__).resolve().parent.parent
    sample_dir = project_root / "data" / "samples"
    out_dir = project_root / "data" / "processed"
    
    # Ensure the output directory exists
    out_dir.mkdir(parents=True, exist_ok=True)

    # 2. Look for the first image file in data/samples
    supported = ["*.png", "*.jpg", "*.jpeg"]
    files = []
    for pattern in supported:
        files.extend(sample_dir.glob(pattern))

    if not files:
        print(f"No sample images found in {sample_dir}")
        return

    # 3. Process the first image found
    img_path = files[0]
    print(f"Processing: {img_path.name}")
    
    with Image.open(img_path) as img:
        # Create a simple preview (grayscale to mimic X-ray style)
        preview = img.convert("L") 
        save_path = out_dir / "preview.png"
        preview.save(save_path)
        print(f"✅ Preview saved to: {save_path}")

if __name__ == "__main__":
    main()