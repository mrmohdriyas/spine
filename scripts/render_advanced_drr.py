import os
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def prepare_advanced_synthetic_verse(out_dir, num_samples=15):
    """Generates synthetic CTs with soft tissue to test physical rendering."""
    os.makedirs(out_dir, exist_ok=True)
    shape = (256, 256, 128)
    labels = [20, 21, 22, 23, 24]
    z_centers = [100, 80, 60, 40, 20]
    
    for idx in range(1, num_samples + 1):
        # Air background
        ct_data = np.full(shape, -1000, dtype=np.float32)
        seg_data = np.zeros(shape, dtype=np.uint8)
        
        # Soft tissue cylinder (water ~ 0 HU)
        x, y, z = np.ogrid[0:shape[0], 0:shape[1], 0:shape[2]]
        cx = 128 + np.random.randint(-10, 10)
        cy = 128 + np.random.randint(-10, 10)
        
        # Draw soft tissue body (large cylinder) along Z axis
        body_mask = (((x - cx)**2 / 70**2) + ((y - cy)**2 / 100**2) <= 1.0) & (z > -1)
        ct_data[body_mask] = 0 + np.random.randn(*ct_data[body_mask].shape) * 20 # 0 HU + noise
        
        for i, label in enumerate(labels):
            cz = z_centers[i] + np.random.randint(-5, 5)
            mask = ((x - cx)**2 / 20**2) + ((y - cy)**2 / 30**2) + ((z - cz)**2 / 10**2) <= 1.0
            ct_data[mask] = 400 + np.random.randint(-50, 50)
            seg_data[mask] = label
            
        affine = np.eye(4)
        nib.save(nib.Nifti1Image(ct_data, affine), os.path.join(out_dir, f"verse_{idx:03d}_ct.nii.gz"))
        nib.save(nib.Nifti1Image(seg_data, affine), os.path.join(out_dir, f"verse_{idx:03d}_seg.nii.gz"))
        print(f"Generated advanced synthetic VerSe sample {idx}")

def render_advanced_drr():
    data_dir = r"c:\projects\spine\data\verse_samples_v2"
    out_dir = r"c:\projects\spine\outputs\phase11_domain_adaptation"
    dataset_dir = r"c:\projects\spine\data\verse_drr_v2"
    
    # 1. Generate Advanced Synthetic Data
    prepare_advanced_synthetic_verse(data_dir, num_samples=15)
        
    for split in ['train', 'val', 'test', 'annotations']:
        os.makedirs(os.path.join(dataset_dir, split), exist_ok=True)
        
    ct_files = sorted(os.listdir(data_dir))
    ct_files = [f for f in ct_files if f.endswith("_ct.nii.gz")]
    
    report_data = []
    all_annotations = []
    
    for idx, ct_filename in enumerate(ct_files):
        scan_id = ct_filename.split("_ct")[0]
        ct_path = os.path.join(data_dir, ct_filename)
        seg_path = os.path.join(data_dir, ct_filename.replace("_ct", "_seg"))
        
        ct_img = nib.load(ct_path)
        seg_img = nib.load(seg_path)
        ct_data = ct_img.get_fdata()
        seg_data = seg_img.get_fdata()
        
        spacing = ct_img.header.get_zooms()
        
        # 2. Physics-Based Rendering (Beer-Lambert Approximation)
        # Mu_water for ~100keV X-rays is ~0.17 cm^-1
        mu_water = 0.17
        
        # Convert HU to attenuation coefficient (mu)
        # HU = 1000 * (mu - mu_water) / mu_water
        # mu = (HU / 1000 + 1) * mu_water
        ct_data_clipped = np.clip(ct_data, -1000, 3000)
        mu_volume = (ct_data_clipped / 1000.0 + 1.0) * mu_water
        
        # Projection along Y-axis (AP view). We sum (mu * dy)
        # spacing[1] is the voxel size along Y in mm. Convert to cm.
        dy_cm = spacing[1] / 10.0
        
        # The sum of mu along the ray gives the line integral of attenuation: \int \mu dy
        ray_integral = np.sum(mu_volume * dy_cm, axis=1)
        
        # Standard X-rays display -log(I/I0), which is exactly the ray integral.
        drr = ray_integral
        
        # Normalize DRR for 8-bit saving (mimicking windowing of X-rays)
        drr_norm = (drr - np.percentile(drr, 1)) / (np.percentile(drr, 99) - np.percentile(drr, 1) + 1e-8)
        drr_norm = np.clip(drr_norm, 0, 1)
        
        # Invert so bone is white, air is black (standard radiographic display)
        # Actually, higher ray_integral = higher attenuation = whiter on standard radiograph.
        # So we don't need to invert if we want high attenuation to be white.
        drr_img = (drr_norm * 255).astype(np.uint8).T
        
        # Find annotations
        unique_labels = [l for l in np.unique(seg_data) if l > 0]
        
        for lbl in unique_labels:
            mask = (seg_data == lbl)
            coords = np.argwhere(mask)
            if len(coords) == 0: continue
            
            x_min, _, z_min = coords.min(axis=0)
            x_max, _, z_max = coords.max(axis=0)
            
            center_x = (x_min + x_max) / 2.0
            center_y = (z_min + z_max) / 2.0
            
            ann = {
                "scan_id": scan_id,
                "label_id": int(lbl),
                "x_min": float(x_min),
                "y_min": float(z_min),
                "x_max": float(x_max),
                "y_max": float(z_max),
                "center_x": float(center_x),
                "center_y": float(center_y)
            }
            all_annotations.append(ann)
            
        report_data.append({
            "scan_id": scan_id,
            "projection_type": "beer-lambert-log",
            "mu_water": mu_water,
            "voxel_spacing_y_mm": spacing[1]
        })
        
        if idx < 10:
            split = 'train'
        elif idx < 13:
            split = 'val'
        else:
            split = 'test'
            
        plt.imsave(os.path.join(dataset_dir, split, f"{scan_id}.png"), drr_img, cmap='gray', origin='lower')
        
    pd.DataFrame(report_data).to_csv(os.path.join(out_dir, "drr_metadata.csv"), index=False)
    pd.DataFrame(all_annotations).to_csv(os.path.join(dataset_dir, "annotations", "drr_annotations.csv"), index=False)
    print("Advanced DRR V2 Generation Complete.")

if __name__ == "__main__":
    render_advanced_drr()
