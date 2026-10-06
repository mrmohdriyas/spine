import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

def inspect_verse():
    data_dir = r"c:\projects\spine\data\verse_samples"
    out_dir = r"c:\projects\spine\outputs\phase10_verse_prototype"
    os.makedirs(out_dir, exist_ok=True)
    
    seg_path = os.path.join(data_dir, "verse_001_seg.nii.gz")
    ct_path = os.path.join(data_dir, "verse_001_ct.nii.gz")
    
    seg_img = nib.load(seg_path)
    ct_img = nib.load(ct_path)
    
    seg_data = seg_img.get_fdata()
    ct_data = ct_img.get_fdata()
    
    print(f"CT Shape: {ct_data.shape}")
    print(f"CT Spacing: {ct_img.header.get_zooms()}")
    
    unique_labels = np.unique(seg_data)
    print(f"Unique labels found: {unique_labels}")
    
    # VerSe Label Mapping
    # C1-C7: 1-7
    # T1-T12: 8-19
    # L1-L5: 20-24
    
    mapping = {}
    for lbl in unique_labels:
        lbl = int(lbl)
        if lbl == 0:
            continue
        elif 1 <= lbl <= 7:
            mapping[lbl] = f"C{lbl}"
        elif 8 <= lbl <= 19:
            mapping[lbl] = f"T{lbl - 7}"
        elif 20 <= lbl <= 24:
            mapping[lbl] = f"L{lbl - 19}"
        else:
            mapping[lbl] = f"OTHER_{lbl}"
            
    print("Detected Vertebrae:")
    for k, v in mapping.items():
        print(f"  Label {k} -> {v}")
        
    # Visualization
    # Find a sagittal slice (X-axis in our H,W,D format would be index 0 or 1. Let's use cx ~ 128)
    cx = ct_data.shape[0] // 2
    
    # We want a slice that shows the spine. Since our synth spine is at cx=128
    ct_slice = ct_data[cx, :, :]
    seg_slice = seg_data[cx, :, :]
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(ct_slice.T, cmap='gray', origin='lower')
    plt.title("CT Slice (Sagittal)")
    
    plt.subplot(1, 2, 2)
    plt.imshow(ct_slice.T, cmap='gray', origin='lower')
    plt.imshow(seg_slice.T, cmap='jet', alpha=0.5, origin='lower')
    plt.title("CT + Segmentation Labels")
    
    plt.savefig(os.path.join(out_dir, "vertebral_label_check.png"))
    plt.close()
    
    print(f"Saved visualization to {os.path.join(out_dir, 'vertebral_label_check.png')}")

if __name__ == "__main__":
    inspect_verse()
