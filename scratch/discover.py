import os
import json

dataset_path = r"C:\Users\Mohamed Riyas\.cache\kagglehub\datasets\siddhale937e92739\annotated-medical-image-dataset-for-spinal-lesions\versions\1\physionet.org\files\vindr-spinexr\1.0.0"

print("Scanning:", dataset_path)
if not os.path.exists(dataset_path):
    print("Path does not exist!")

def scan_dir(path, level=0, max_level=5):
    if level > max_level: return
    try:
        entries = os.listdir(path)
        for entry in entries[:10]:
            full_path = os.path.join(path, entry)
            print("  " * level + "|--", entry)
            if os.path.isdir(full_path):
                scan_dir(full_path, level + 1, max_level)
        if len(entries) > 10:
            print("  " * level + "|--", f"... and {len(entries)-10} more")
    except Exception as e:
        print("Error:", e)

scan_dir(dataset_path)
