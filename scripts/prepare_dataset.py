import os
import random
import json
import pandas as pd
import pydicom
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch

# --- CONFIGURATION ---
DATASET_PATH = r"C:\Users\Mohamed Riyas\.cache\kagglehub\datasets\siddhale937e92739\annotated-medical-image-dataset-for-spinal-lesions\versions\1\physionet.org\files\vindr-spinexr\1.0.0"
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

def get_image_dims(dicom_path):
    try:
        ds = pydicom.dcmread(dicom_path, stop_before_pixels=True)
        return ds.Rows, ds.Columns
    except:
        return None, None

def main():
    print("Starting dataset preparation...")
    
    # 1. Paths
    train_csv = os.path.join(DATASET_PATH, "annotations", "train.csv")
    test_csv = os.path.join(DATASET_PATH, "annotations", "test.csv")
    train_images_dir = os.path.join(DATASET_PATH, "train_images")
    test_images_dir = os.path.join(DATASET_PATH, "test_images")
    
    df_train_orig = pd.read_csv(train_csv)
    df_test_orig = pd.read_csv(test_csv)
    
    # 2. Class Mapping
    classes = set(df_train_orig['lesion_type'].unique()) | set(df_test_orig['lesion_type'].unique())
    classes = sorted(list(classes))
    # Ensure "No finding" is 0 if it exists
    if "No finding" in classes:
        classes.remove("No finding")
        classes.insert(0, "No finding")
        
    class_mapping = {str(i): c for i, c in enumerate(classes)}
    inverse_class_mapping = {c: i for i, c in enumerate(classes)}
    
    os.makedirs(r"c:\projects\spine\data", exist_ok=True)
    with open(r"c:\projects\spine\data\class_mapping.json", "w") as f:
        json.dump(class_mapping, f, indent=2)
        
    # 3. Clean and Validate
    def process_df(df, images_dir, split_name):
        records = []
        invalid = []
        
        # We need unique image paths to get dims quickly. Cache them.
        img_dims = {}
        for img_id in df['image_id'].unique():
            path = os.path.join(images_dir, f"{img_id}.dicom")
            img_dims[img_id] = {'path': path, 'rows': None, 'cols': None}
        
        for idx, row in df.iterrows():
            img_id = row['image_id']
            path_info = img_dims[img_id]
            path = path_info['path']
            
            if not os.path.exists(path):
                continue
                
            # Read dims if not cached and needed
            if path_info['rows'] is None and pd.notna(row['xmin']):
                r, c = get_image_dims(path)
                path_info['rows'] = r
                path_info['cols'] = c
            
            is_valid = True
            reason = ""
            
            x_min, y_min, x_max, y_max = row['xmin'], row['ymin'], row['xmax'], row['ymax']
            label = row['lesion_type']
            
            if pd.notna(x_min):
                x_min, y_min, x_max, y_max = float(x_min), float(y_min), float(x_max), float(y_max)
                if x_min >= x_max:
                    is_valid, reason = False, "x_min >= x_max"
                elif y_min >= y_max:
                    is_valid, reason = False, "y_min >= y_max"
                elif x_min < 0 or y_min < 0:
                    is_valid, reason = False, "Negative coordinates"
                elif path_info['rows'] is not None and path_info['cols'] is not None:
                    if x_max > path_info['cols'] or y_max > path_info['rows']:
                        is_valid, reason = False, "Out of bounds"
            
            record = {
                'image_id': img_id,
                'image_path': path,
                'label': label,
                'label_id': inverse_class_mapping[label],
                'x_min': x_min,
                'y_min': y_min,
                'x_max': x_max,
                'y_max': y_max,
                'split': split_name
            }
            
            if is_valid:
                records.append(record)
            else:
                record['reason'] = reason
                invalid.append(record)
                
        return pd.DataFrame(records), pd.DataFrame(invalid)
    
    print("Processing train dataset...")
    df_train_valid, df_train_invalid = process_df(df_train_orig, train_images_dir, "train")
    print("Processing test dataset...")
    df_test_valid, df_test_invalid = process_df(df_test_orig, test_images_dir, "test")
    
    # 4. Train / Val split (from train)
    unique_train_images = list(df_train_valid['image_id'].unique())
    random.shuffle(unique_train_images)
    
    val_size = int(len(unique_train_images) * 0.15)
    val_images = set(unique_train_images[:val_size])
    
    df_train_final = df_train_valid[~df_train_valid['image_id'].isin(val_images)].copy()
    df_val_final = df_train_valid[df_train_valid['image_id'].isin(val_images)].copy()
    df_val_final['split'] = 'val'
    
    # 5. Save manifests
    os.makedirs(r"c:\projects\spine\data\manifests", exist_ok=True)
    df_train_final.to_csv(r"c:\projects\spine\data\manifests\train.csv", index=False)
    df_val_final.to_csv(r"c:\projects\spine\data\manifests\val.csv", index=False)
    df_test_valid.to_csv(r"c:\projects\spine\data\manifests\test.csv", index=False)
    
    # Invalid
    df_invalid = pd.concat([df_train_invalid, df_test_invalid])
    os.makedirs(r"c:\projects\spine\docs", exist_ok=True)
    if not df_invalid.empty:
        df_invalid.to_csv(r"c:\projects\spine\docs\invalid_annotations.csv", index=False)
        print(f"Found {len(df_invalid)} invalid annotations.")
    
    # 6. Basic Statistics
    stats_md = f"""# Phase 2 — Dataset Statistics

- Total images: {len(df_train_final['image_id'].unique()) + len(df_val_final['image_id'].unique()) + len(df_test_valid['image_id'].unique())}
- Training images: {len(df_train_final['image_id'].unique())}
- Validation images: {len(df_val_final['image_id'].unique())}
- Test images: {len(df_test_valid['image_id'].unique())}
- Total annotations: {len(df_train_final) + len(df_val_final) + len(df_test_valid)}
- Invalid annotations: {len(df_invalid)}

## Class Distribution

| Class | Total Annotations |
| --- | --- |
"""
    all_valid = pd.concat([df_train_final, df_val_final, df_test_valid])
    counts = all_valid['label'].value_counts()
    for c in classes:
        stats_md += f"| {c} | {counts.get(c, 0)} |\n"
        
    with open(r"c:\projects\spine\docs\PHASE_2_DATASET_STATISTICS.md", "w") as f:
        f.write(stats_md)
        
    # 7. Visualization
    print("Generating visualizations...")
    samples_dir = r"c:\projects\spine\docs\phase2_samples"
    os.makedirs(samples_dir, exist_ok=True)
    
    sample_images = random.sample(list(all_valid['image_id'].unique()), min(15, len(all_valid['image_id'].unique())))
    for img_id in sample_images:
        group = all_valid[all_valid['image_id'] == img_id]
        img_path = group.iloc[0]['image_path']
        
        try:
            ds = pydicom.dcmread(img_path)
            img = ds.pixel_array
            
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(img, cmap='gray')
            for _, row in group.iterrows():
                if pd.notna(row['x_min']):
                    rect = patches.Rectangle((row['x_min'], row['y_min']), 
                                             row['x_max'] - row['x_min'], 
                                             row['y_max'] - row['y_min'], 
                                             linewidth=2, edgecolor='r', facecolor='none')
                    ax.add_patch(rect)
                    ax.text(row['x_min'], max(0, row['y_min']-10), row['label'], color='r', fontsize=8, backgroundcolor='white')
            
            plt.title(f"Image: {img_id}")
            plt.axis('off')
            plt.savefig(os.path.join(samples_dir, f"{img_id}.png"))
            plt.close()
        except Exception as e:
            print(f"Error visualizing {img_id}: {e}")

    # 8. Verify Everything
    print("\nVerifying...")
    try:
        import sys
        sys.path.insert(0, r"c:\projects\spine")
        from src.dataset.spine_dataset import SpineDataset
        from src.dataset.transforms import BasicTransform
        
        tf = BasicTransform()
        ds = SpineDataset(r"c:\projects\spine\data\manifests\train.csv", transforms=tf)
        img, target = ds[0]
        
        loader_status = "PASS"
        sample_status = "PASS" if isinstance(img, torch.Tensor) else "FAIL"
    except Exception as e:
        print(f"Loader Error: {e}")
        loader_status = "FAIL"
        sample_status = "FAIL"
        
    print("\nPHASE 2 COMPLETE\n")
    print(f"Images:\nTrain: {len(df_train_final['image_id'].unique())}\nValidation: {len(df_val_final['image_id'].unique())}\nTest: {len(df_test_valid['image_id'].unique())}\n")
    print(f"Annotations: {len(all_valid)}\nClasses: {len(classes)}\n")
    print(f"Invalid annotations: {len(df_invalid)}\n")
    print(f"Dataset loader:\n{loader_status}\n")
    print(f"Sample loading:\n{sample_status}\n")
    print("Files created:")
    print("- data/manifests/train.csv")
    print("- data/manifests/val.csv")
    print("- data/manifests/test.csv")
    print("- data/class_mapping.json")
    print("- docs/PHASE_2_DATASET_STATISTICS.md")
    print("- src/dataset/spine_dataset.py")
    print("- src/dataset/transforms.py")

if __name__ == "__main__":
    main()
