import os
import pandas as pd
import pydicom

def check_integrity():
    splits = ['train', 'val', 'test']
    invalid_records = []
    
    total_checked = 0
    valid_count = 0
    missing_files = 0
    missing_pixel_data = 0
    other_errors = 0
    
    for split in splits:
        manifest_path = rf"c:\projects\spine\data\manifests\{split}.csv"
        if not os.path.exists(manifest_path):
            continue
            
        df = pd.read_csv(manifest_path)
        unique_images = df[['image_id', 'image_path']].drop_duplicates()
        
        for _, row in unique_images.iterrows():
            total_checked += 1
            path = row['image_path']
            image_id = row['image_id']
            
            if not os.path.exists(path):
                missing_files += 1
                invalid_records.append({
                    "image_id": image_id,
                    "file_path": path,
                    "split": split,
                    "reason": "File not found"
                })
                continue
                
            try:
                # Lightweight check: do not decode pixels
                ds = pydicom.dcmread(path, defer_size=1024) # just read header
                
                # Some versions of pydicom may not load it into dict if we stop before pixels.
                # So we just load it but do not access pixel_array. We check 'PixelData' tag.
                if 'PixelData' not in ds:
                    missing_pixel_data += 1
                    invalid_records.append({
                        "image_id": image_id,
                        "file_path": path,
                        "split": split,
                        "reason": "Missing Pixel Data"
                    })
                else:
                    valid_count += 1
            except Exception as e:
                other_errors += 1
                invalid_records.append({
                    "image_id": image_id,
                    "file_path": path,
                    "split": split,
                    "reason": f"Read error: {e}"
                })
                
    # Write CSV
    if invalid_records:
        invalid_df = pd.DataFrame(invalid_records)
        invalid_df.to_csv(r"c:\projects\spine\docs\invalid_dicom_files.csv", index=False)
    else:
        pd.DataFrame(columns=["image_id", "file_path", "split", "reason"]).to_csv(r"c:\projects\spine\docs\invalid_dicom_files.csv", index=False)
        
    # Write Report
    report = f"""# Dataset Integrity Report

Total images checked: {total_checked}
Valid images: {valid_count}
Invalid images: {len(invalid_records)}
Missing images: {missing_files}
Missing Pixel Data: {missing_pixel_data}
Other read errors: {other_errors}
"""
    with open(r"c:\projects\spine\docs\dataset_integrity_report.md", "w") as f:
        f.write(report)
        
    print("Integrity check complete.")
    print(f"Invalid files found: {len(invalid_records)}")

if __name__ == '__main__':
    check_integrity()
