import os
import json
import random
import pandas as pd
from collections import Counter

def validate_and_split():
    data_dir = r"c:\projects\spine\data\vindr_vertebral_annotations"
    raw_path = os.path.join(data_dir, "raw_annotations.json")
    
    with open(raw_path, 'r') as f:
        data = json.load(f)
        
    valid_data = []
    issues = []
    
    for item in data:
        img_id = item['image_id']
        anns = item['annotations']
        
        # Check for empty
        if len(anns) == 0:
            issues.append({"image_id": img_id, "issue": "Empty annotations"})
            continue
            
        # Check for duplicate labels
        labels = [a['vertebra'] for a in anns]
        counts = Counter(labels)
        duplicates = [k for k, v in counts.items() if v > 1]
        
        if duplicates:
            issues.append({"image_id": img_id, "issue": f"Duplicate labels: {duplicates}"})
            # To fix the dummy data: keep only the first occurrence of each label
            seen = set()
            new_anns = []
            for a in anns:
                if a['vertebra'] not in seen:
                    seen.add(a['vertebra'])
                    new_anns.append(a)
            anns = new_anns
            
        # Validate boxes
        valid_anns = []
        for a in anns:
            x1, y1, x2, y2 = a['bbox']
            if x2 <= x1 or y2 <= y1:
                issues.append({"image_id": img_id, "issue": f"Invalid box: {a['bbox']}"})
                continue
            if a['vertebra'] not in ['L1', 'L2', 'L3', 'L4', 'L5']:
                issues.append({"image_id": img_id, "issue": f"Invalid label: {a['vertebra']}"})
                continue
            valid_anns.append(a)
            
        if len(valid_anns) > 0:
            valid_data.append({"image_id": img_id, "annotations": valid_anns})
            
    # Save QC Report
    os.makedirs(r"c:\projects\spine\outputs\phase13_real_xray_validation", exist_ok=True)
    if issues:
        pd.DataFrame(issues).to_csv(r"c:\projects\spine\outputs\phase13_real_xray_validation\annotation_quality.csv", index=False)
        
    # Shuffle and Split (70, 15, 15)
    random.seed(42)
    random.shuffle(valid_data)
    
    n = len(valid_data)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)
    
    train_data = valid_data[:n_train]
    val_data = valid_data[n_train:n_train+n_val]
    test_data = valid_data[n_train+n_val:]
    
    with open(os.path.join(data_dir, "train.json"), "w") as f: json.dump(train_data, f, indent=4)
    with open(os.path.join(data_dir, "val.json"), "w") as f: json.dump(val_data, f, indent=4)
    with open(os.path.join(data_dir, "test.json"), "w") as f: json.dump(test_data, f, indent=4)
    
    print(f"Validated {len(valid_data)} images.")
    print(f"Splits -> Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

if __name__ == "__main__":
    validate_and_split()
