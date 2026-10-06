import os
import glob
import json
import pandas as pd
import pydicom
import matplotlib.pyplot as plt

dataset_path = r"C:\Users\Mohamed Riyas\.cache\kagglehub\datasets\siddhale937e92739\annotated-medical-image-dataset-for-spinal-lesions\versions\1\physionet.org\files\vindr-spinexr\1.0.0"

def audit_dataset():
    print("Starting dataset audit...")
    summary = {
        "dataset_path": dataset_path,
        "total_files": 0,
        "total_images": 0,
        "image_formats": [],
        "image_dimensions": [],
        "channels": [],
        "labels": [],
        "classes": {},
        "annotations": [],
        "vertebral_information": {
            "localization": "Not available",
            "segmentation": "Not available",
            "identification": "Not available"
        }
    }

    # 1. Inspect file structure
    all_files = glob.glob(os.path.join(dataset_path, '**', '*'), recursive=True)
    files_only = [f for f in all_files if os.path.isfile(f)]
    summary["total_files"] = len(files_only)
    
    extensions = {}
    for f in files_only:
        ext = os.path.splitext(f)[1].lower()
        extensions[ext] = extensions.get(ext, 0) + 1
    print(f"File extensions: {extensions}")

    # 2. Images
    dicom_files = [f for f in files_only if f.endswith('.dicom') or f.endswith('.dcm')]
    summary["total_images"] = len(dicom_files)
    if dicom_files:
        summary["image_formats"].append("DICOM")
        # Sample images
        dims = set()
        channels = set()
        for df in dicom_files[:10]:
            try:
                ds = pydicom.dcmread(df)
                dims.add((ds.Rows, ds.Columns))
                channels.add(getattr(ds, 'SamplesPerPixel', 1))
            except Exception as e:
                print(f"Error reading {df}: {e}")
        summary["image_dimensions"] = [f"{w}x{h}" for h, w in dims]
        summary["channels"] = list(channels)

    # 3. Annotations & Labels
    train_csv = os.path.join(dataset_path, "annotations", "train.csv")
    test_csv = os.path.join(dataset_path, "annotations", "test.csv")
    
    if os.path.exists(train_csv):
        summary["annotations"].append("train.csv")
        df_train = pd.read_csv(train_csv)
        if "lesion_type" in df_train.columns:
            summary["labels"].append("lesion_type (bounding boxes)")
            counts = df_train["lesion_type"].value_counts().to_dict()
            summary["classes"] = counts

            # Check for vertebral info
            # Usually vertebral labels might be like "V1", "C2", etc., or 'lesion_type' could just be 'Other lesion', 'Osteophyte'.
            # We will inspect the classes.
            lesion_types = list(counts.keys())
            if any("T" in lt or "C" in lt or "L" in lt or "vertebra" in lt.lower() for lt in lesion_types):
                 # Let's be conservative unless explicit
                 pass

    if os.path.exists(test_csv):
        summary["annotations"].append("test.csv")

    # Generate Visualization
    if dicom_files and os.path.exists(train_csv):
        try:
            sample_img_path = dicom_files[0]
            ds = pydicom.dcmread(sample_img_path)
            img = ds.pixel_array
            
            image_id = os.path.basename(sample_img_path).replace('.dicom', '')
            sample_anno = df_train[df_train['image_id'] == image_id]
            
            plt.figure(figsize=(8, 8))
            plt.imshow(img, cmap='gray')
            for _, row in sample_anno.iterrows():
                if pd.notna(row['xmin']):
                    rect = plt.Rectangle((row['xmin'], row['ymin']), 
                                         row['xmax'] - row['xmin'], 
                                         row['ymax'] - row['ymin'], 
                                         fill=False, color='red', linewidth=2)
                    plt.gca().add_patch(rect)
                    plt.text(row['xmin'], row['ymin'], row['lesion_type'], color='red')
                    
            plt.title(f"Sample: {image_id}")
            os.makedirs(r"c:\projects\spine\docs", exist_ok=True)
            vis_path = r"c:\projects\spine\docs\sample_visualization.png"
            plt.savefig(vis_path)
            plt.close()
            print(f"Saved visualization to {vis_path}")
        except Exception as e:
            print(f"Visualization error: {e}")

    # Save summary
    os.makedirs(r"c:\projects\spine\docs", exist_ok=True)
    with open(r"c:\projects\spine\docs\dataset_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Saved dataset_summary.json")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    audit_dataset()
