from pathlib import Path
import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid, SecondaryCaptureImageStorage

def main():
    root = Path(__file__).resolve().parents[1]
    raw_dir = root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_path = raw_dir / "sample.dcm"

    rows, cols = 512, 512
    y, x = np.ogrid[:rows, :cols]
    center_y, center_x = rows // 2, cols // 2
    r = np.sqrt((y - center_y) ** 2 + (x - center_x) ** 2)

    img = 4000 - (r * 6)
    img = np.clip(img, 0, 4095)

    spot = np.exp(-(((x - (center_x + 90)) ** 2 + (y - (center_y - 40)) ** 2) / (2 * (35 ** 2))))
    img = img + spot * 1500
    img = np.clip(img, 0, 4095).astype(np.uint16)

    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    ds = FileDataset(str(out_path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.is_little_endian = True
    ds.is_implicit_VR = False

    ds.SOPClassUID = SecondaryCaptureImageStorage
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()

    ds.Modality = "OT"
    ds.PatientName = "Test^Sample"
    ds.PatientID = "SAMPLE001"

    ds.Rows = rows
    ds.Columns = cols
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 12
    ds.HighBit = 11
    ds.PixelRepresentation = 0
    ds.PixelData = img.tobytes()

    ds.save_as(str(out_path))
    print(f"Created DICOM: {out_path}")

if __name__ == "__main__":
    main()
