import kagglehub

# Download latest version
print("Starting download...")
path = kagglehub.dataset_download("siddhale937e92739/annotated-medical-image-dataset-for-spinal-lesions")

print("Path to dataset files:", path)
