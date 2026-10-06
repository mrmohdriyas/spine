import os
import pydicom
import numpy as np
import pandas as pd
from PIL import Image

def select_vindr_images():
    manifest_path = r"c:\projects\spine\data\manifests\train.csv"
    dicom_dir = r"C:\Users\Mohamed Riyas\.cache\kagglehub\datasets\siddhale937e92739\annotated-medical-image-dataset-for-spinal-lesions\versions\1\physionet.org\files\vindr-spinexr\1.0.0\train_images"
    
    out_dir = r"c:\projects\spine\data\vindr_vertebral_annotations"
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    
    df = pd.read_csv(manifest_path)
    
    # Select 30 unique images. To get variation, we just take the first 30 unique ones.
    unique_img_ids = df['image_id'].unique()[:30]
    
    selected_data = []
    
    for img_id in unique_img_ids:
        dicom_path = os.path.join(dicom_dir, f"{img_id}.dicom")
        if not os.path.exists(dicom_path):
            continue
            
        ds = pydicom.dcmread(dicom_path)
        pixel_array = ds.pixel_array.astype(float)
        pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min() + 1e-8) * 255
        pixel_array = pixel_array.astype(np.uint8)
        
        img = Image.fromarray(pixel_array).convert("RGB")
        img.save(os.path.join(img_dir, f"{img_id}.png"))
        
        # Get pathology labels
        pathologies = df[df['image_id'] == img_id]['label'].tolist()
        
        selected_data.append({
            "image_id": img_id,
            "selection_reason": "Representative sample for manual annotation prototype",
            "pathology_labels": ";".join(set(pathologies))
        })
        
    pd.DataFrame(selected_data).to_csv(os.path.join(out_dir, "selected_images.csv"), index=False)
    print(f"Selected {len(selected_data)} images for annotation.")

if __name__ == "__main__":
    select_vindr_images()
