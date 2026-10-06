import os
import numpy as np
import nibabel as nib

def generate_synthetic_verse():
    out_dir = r"c:\projects\spine\data\verse_samples"
    os.makedirs(out_dir, exist_ok=True)
    
    # Create a 3D volume (Z, Y, X) -> Let's use (128, 256, 128)
    # Medical images often have shapes like (Depth, Height, Width) or (H, W, D)
    shape = (256, 256, 128)
    labels = [20, 21, 22, 23, 24]
    z_centers = [100, 80, 60, 40, 20]
    
    for idx in range(1, 6):
        ct_data = np.full(shape, -1000, dtype=np.float32) # Air in Hounsfield Units
        seg_data = np.zeros(shape, dtype=np.uint8)
        
        # Add some random translation to each scan so they aren't perfectly identical
        cx = 128 + np.random.randint(-10, 10)
        cy = 128 + np.random.randint(-10, 10)
        
        x, y, z = np.ogrid[0:shape[0], 0:shape[1], 0:shape[2]]
        
        for i, label in enumerate(labels):
            cz = z_centers[i] + np.random.randint(-5, 5)
            
            mask = ((x - cx)**2 / 20**2) + ((y - cy)**2 / 30**2) + ((z - cz)**2 / 10**2) <= 1.0
            ct_data[mask] = 400 + np.random.randint(-50, 50)
            seg_data[mask] = label
                            
        affine = np.eye(4)
        ct_img = nib.Nifti1Image(ct_data, affine)
        seg_img = nib.Nifti1Image(seg_data, affine)
        
        nib.save(ct_img, os.path.join(out_dir, f"verse_{idx:03d}_ct.nii.gz"))
        nib.save(seg_img, os.path.join(out_dir, f"verse_{idx:03d}_seg.nii.gz"))
        print(f"Generated synthetic VerSe sample {idx}")
    
    print(f"Generated synthetic VerSe sample at {out_dir}")

if __name__ == "__main__":
    generate_synthetic_verse()
