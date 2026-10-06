import os
import glob
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json

def generate_drrs():
    data_dir = r"c:\projects\spine\data\verse_samples"
    out_dir = r"c:\projects\spine\outputs\phase10_verse_prototype"
    dataset_dir = r"c:\projects\spine\data\verse_drr"
    
    os.makedirs(os.path.join(out_dir, "drr_visualizations"), exist_ok=True)
    for split in ['train', 'val', 'test', 'annotations']:
        os.makedirs(os.path.join(dataset_dir, split), exist_ok=True)
        
    ct_files = sorted(glob.glob(os.path.join(data_dir, "*_ct.nii.gz")))
    
    report_data = []
    all_annotations = []
    
    for idx, ct_path in enumerate(ct_files):
        scan_id = os.path.basename(ct_path).split("_ct")[0]
        seg_path = ct_path.replace("_ct", "_seg")
        
        ct_data = nib.load(ct_path).get_fdata()
        seg_data = nib.load(seg_path).get_fdata()
        
        # 3D to 2D Projection: Sum along Y-axis (Coronal/Sagittal projection depending on axes)
        # Let's project along Y-axis to get an AP/PA view.
        # Shape: (X, Y, Z) -> (X, Z)
        drr = np.sum(ct_data, axis=1)
        
        # Normalize DRR for saving as image
        drr_norm = (drr - drr.min()) / (drr.max() - drr.min() + 1e-8)
        drr_img = (drr_norm * 255).astype(np.uint8)
        
        # We need Z to be the vertical axis in our image. Currently it's (X, Z).
        # Let's transpose to (Z, X) so Z is height.
        drr_img = drr_img.T
        
        # Find annotations
        unique_labels = np.unique(seg_data)
        unique_labels = [l for l in unique_labels if l > 0]
        
        centroids = []
        annotations = []
        
        fig, ax = plt.subplots(1, figsize=(8, 8))
        ax.imshow(drr_img, cmap='gray', origin='lower')
        
        for lbl in unique_labels:
            mask = (seg_data == lbl)
            # Find coordinates of the mask
            coords = np.argwhere(mask) # (N, 3) -> [x, y, z]
            
            if len(coords) == 0:
                continue
                
            x_min, _, z_min = coords.min(axis=0)
            x_max, _, z_max = coords.max(axis=0)
            
            # Since we transposed to (Z, X) for the image:
            # X-axis on image = X coordinate
            # Y-axis on image = Z coordinate
            img_x_min = x_min
            img_x_max = x_max
            img_y_min = z_min
            img_y_max = z_max
            
            center_x = (img_x_min + img_x_max) / 2.0
            center_y = (img_y_min + img_y_max) / 2.0
            
            centroids.append((lbl, center_y))
            
            ann = {
                "scan_id": scan_id,
                "label_id": int(lbl),
                "x_min": float(img_x_min),
                "y_min": float(img_y_min),
                "x_max": float(img_x_max),
                "y_max": float(img_y_max),
                "center_x": float(center_x),
                "center_y": float(center_y)
            }
            annotations.append(ann)
            all_annotations.append(ann)
            
            # Draw box
            rect = patches.Rectangle((img_x_min, img_y_min), img_x_max - img_x_min, img_y_max - img_y_min,
                                     linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(img_x_min, img_y_max, f"L{int(lbl)-19}", color='red', fontsize=8)
            
        plt.title(f"DRR Projection: {scan_id}")
        plt.savefig(os.path.join(out_dir, "drr_visualizations", f"{scan_id}_overlay.png"))
        plt.close()
        
        # Check Anatomical Order (Y should decrease from L1 to L5 if origin is lower, meaning L1 is superior/higher Z)
        # Wait, if Z=100 is L1 and Z=20 is L5. Then L1 has higher Y than L5.
        centroids.sort(key=lambda x: x[0]) # Sort by label (20->24)
        ordering_violations = 0
        for i in range(len(centroids) - 1):
            if centroids[i][1] < centroids[i+1][1]: # L_i center_y should be > L_{i+1} center_y
                ordering_violations += 1
                
        report_data.append({
            "scan_id": scan_id,
            "num_vertebrae": len(unique_labels),
            "ordering_violations": ordering_violations,
            "missing_labels": 5 - len(unique_labels),
            "duplicate_labels": 0
        })
        
        # Split logic (1-3 train, 4 val, 5 test)
        if idx < 3:
            split = 'train'
        elif idx == 3:
            split = 'val'
        else:
            split = 'test'
            
        plt.imsave(os.path.join(dataset_dir, split, f"{scan_id}.png"), drr_img, cmap='gray', origin='lower')
        
    pd.DataFrame(report_data).to_csv(os.path.join(out_dir, "anatomical_order_report.csv"), index=False)
    pd.DataFrame(all_annotations).to_csv(os.path.join(dataset_dir, "annotations", "drr_annotations.csv"), index=False)
    print("DRR Generation and Validation Complete.")

if __name__ == "__main__":
    generate_drrs()
