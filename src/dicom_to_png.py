from pathlib import Path
import numpy as np
import pydicom
from PIL import Image

def dicom_to_uint8(dcm_path: Path) -> np.ndarray:
    ds = pydicom.dcmread(str(dcm_path))
    arr = ds.pixel_array.astype(np.float32)

    # Apply rescale slope/intercept if present (common in medical images)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    arr = arr * slope + intercept

    # Normalize to 0..255 (uint8) for saving as PNG
    arr = arr - arr.min()
    if arr.max() > 0:
        arr = arr / arr.max()
    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    return arr

def main():
    project_root = Path(__file__).resolve().parents[1]
    raw_dir = project_root / "data" / "raw"
    out_dir = project_root / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    dicoms = list(raw_dir.glob("*.dcm"))
    if not dicoms:
        print("No .dcm files found in data/raw.")
        print("Put ONE DICOM file into data/raw (example: sample.dcm) and run again.")
        return

    dcm_path = dicoms[0]
    img_arr = dicom_to_uint8(dcm_path)

    out_path = out_dir / "dicom_preview.png"
    Image.fromarray(img_arr).save(out_path)

    print(f"Loaded DICOM: {dcm_path.name}")
    print(f"Saved PNG preview to: {out_path}")

if __name__ == "__main__":
    main()
