import os
import glob
import numpy as np
import pandas as pd
import pydicom
import matplotlib.pyplot as plt
from PIL import Image

def characterize_domain_gap():
    drr_dir = r"c:\projects\spine\data\verse_drr\train"
    vindr_manifest = pd.read_csv(r"c:\projects\spine\data\manifests\train.csv")
    
    out_dir = r"c:\projects\spine\outputs\phase11_domain_adaptation"
    os.makedirs(out_dir, exist_ok=True)
    
    drr_files = glob.glob(os.path.join(drr_dir, "*.png"))
    vindr_ids = vindr_manifest['image_id'].unique()[:5] # Small sample
    
    stats = []
    drr_pixels = []
    vindr_pixels = []
    
    # Process DRRs
    drr_sample = None
    for f in drr_files:
        img = np.array(Image.open(f).convert('L'))
        drr_pixels.extend(img.flatten())
        if drr_sample is None:
            drr_sample = img
        stats.append({
            "domain": "VerSe_DRR_V1",
            "file": os.path.basename(f),
            "width": img.shape[1],
            "height": img.shape[0],
            "min": img.min(),
            "max": img.max(),
            "mean": img.mean(),
            "std": img.std(),
            "p25": np.percentile(img, 25),
            "p75": np.percentile(img, 75)
        })
        
    # Process VinDr
    vindr_sample = None
    for vid in vindr_ids:
        dcm_path = fr"C:\Users\Mohamed Riyas\.cache\kagglehub\datasets\siddhale937e92739\annotated-medical-image-dataset-for-spinal-lesions\versions\1\physionet.org\files\vindr-spinexr\1.0.0\train_images\{vid}.dicom"
        if not os.path.exists(dcm_path): continue
        ds = pydicom.dcmread(dcm_path)
        img = ds.pixel_array.astype(float)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8) * 255
        img = img.astype(np.uint8)
        
        vindr_pixels.extend(img.flatten()[::10]) # subsample to save memory
        if vindr_sample is None:
            vindr_sample = img
            
        stats.append({
            "domain": "VinDr_Real",
            "file": f"{vid}.dicom",
            "width": img.shape[1],
            "height": img.shape[0],
            "min": img.min(),
            "max": img.max(),
            "mean": img.mean(),
            "std": img.std(),
            "p25": np.percentile(img, 25),
            "p75": np.percentile(img, 75)
        })
        
    pd.DataFrame(stats).to_csv(os.path.join(out_dir, "intensity_statistics.csv"), index=False)
    
    plt.figure(figsize=(10, 5))
    plt.hist(drr_pixels, bins=50, alpha=0.5, label='VerSe DRR V1', density=True)
    plt.hist(vindr_pixels, bins=50, alpha=0.5, label='VinDr Real', density=True)
    plt.legend()
    plt.title("Intensity Histogram Comparison")
    plt.savefig(os.path.join(out_dir, "intensity_histograms.png"))
    plt.close()
    
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(drr_sample, cmap='gray')
    axs[0].set_title("VerSe DRR V1 (Synthetic)")
    axs[1].imshow(vindr_sample, cmap='gray')
    axs[1].set_title("VinDr Real X-ray")
    plt.savefig(os.path.join(out_dir, "domain_comparison.png"))
    plt.close()
    
    print("Domain gap characterization complete.")

if __name__ == "__main__":
    characterize_domain_gap()
