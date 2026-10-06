import os
import json
import pandas as pd
import pydicom
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2

class SpineDataset(Dataset):
    def __init__(self, manifest_path, class_mapping_path, transforms=None, return_morphology=False, return_graph=False):
        self.manifest_path = manifest_path
        self.df = pd.read_csv(manifest_path)
        
        with open(class_mapping_path, 'r') as f:
            self.class_mapping = json.load(f)
        self.num_classes = len(self.class_mapping)
        
        # We need the inverse mapping to find the index for "No finding"
        self.inverse_mapping = {v: int(k) for k, v in self.class_mapping.items()}
        self.no_finding_idx = self.inverse_mapping.get("No finding", -1)
        
        self.image_groups = self.df.groupby('image_id')
        self.image_ids = list(self.image_groups.groups.keys())
        self.transforms = transforms
        self.return_morphology = return_morphology
        self.return_graph = return_graph

    def __len__(self):
        return len(self.image_ids)

    def _load_dicom(self, path):
        try:
            ds = pydicom.dcmread(path)
            image = ds.pixel_array.astype('float32')
            
            intercept = getattr(ds, 'RescaleIntercept', 0)
            slope = getattr(ds, 'RescaleSlope', 1)
            image = image * slope + intercept
            
            # Normalize to 0-1
            image_min = image.min()
            image_max = image.max()
            if image_max > image_min:
                image = (image - image_min) / (image_max - image_min)
            else:
                image = image - image_min
                
            return image, image.shape[0], image.shape[1]
        except Exception as e:
            print(f"Warning: Failed to load {path} - {e}. Using a blank image.")
            return np.zeros((512, 512), dtype='float32'), 512, 512

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        group = self.image_groups.get_group(image_id)
        
        image_path = group.iloc[0]['image_path']
        image, orig_h, orig_w = self._load_dicom(image_path)
        
        # Create multi-hot target
        target_vec = torch.zeros(self.num_classes, dtype=torch.float32)
        has_pathology = False
        
        for _, row in group.iterrows():
            lbl_id = int(row['label_id'])
            target_vec[lbl_id] = 1.0
            if lbl_id != self.no_finding_idx:
                has_pathology = True
                
        # Handle "No finding" logic
        if self.no_finding_idx != -1:
            if has_pathology:
                target_vec[self.no_finding_idx] = 0.0
            elif target_vec.sum() == 0:
                # If no annotations but it's in the dataset, it's a genuine negative
                # But typically it will have a "No finding" annotation explicitly.
                # Just to be safe:
                target_vec[self.no_finding_idx] = 1.0

        if self.transforms:
            image = self.transforms(image)
            
        if self.return_morphology or self.return_graph:
            morphology_vec = torch.zeros(8, dtype=torch.float32)
            widths, heights, areas, aspect_ratios, center_xs, center_ys = [], [], [], [], [], []
            num_annotations = 0
            
            nodes = []
            
            for _, row in group.iterrows():
                if not pd.isna(row.get('x_min')):
                    xmin = float(row['x_min'])
                    ymin = float(row['y_min'])
                    xmax = float(row['x_max'])
                    ymax = float(row['y_max'])
                    
                    w = (xmax - xmin) / orig_w
                    h = (ymax - ymin) / orig_h
                    area = w * h
                    cx = (xmin + xmax) / (2.0 * orig_w)
                    cy = (ymin + ymax) / (2.0 * orig_h)
                    ar = w / (h + 1e-6)
                    
                    widths.append(w)
                    heights.append(h)
                    areas.append(area)
                    aspect_ratios.append(ar)
                    center_xs.append(cx)
                    center_ys.append(cy)
                    num_annotations += 1
                    
                    nodes.append([cx, cy, w, h, area, ar])
            
            if num_annotations > 0:
                morphology_vec[0] = np.mean(widths)
                morphology_vec[1] = np.mean(heights)
                morphology_vec[2] = np.mean(areas)
                morphology_vec[3] = np.mean(aspect_ratios)
                morphology_vec[4] = np.mean(center_xs)
                morphology_vec[5] = np.mean(center_ys)
                morphology_vec[6] = np.max(areas)
                morphology_vec[7] = num_annotations
                
            if self.return_graph:
                nodes_tensor = torch.tensor(nodes, dtype=torch.float32) if len(nodes) > 0 else torch.zeros((0, 6), dtype=torch.float32)
                edges = []
                edge_feats = []
                
                N = len(nodes)
                if N > 1:
                    for i in range(N):
                        dists = []
                        for j in range(N):
                            if i != j:
                                dx = nodes[j][0] - nodes[i][0]
                                dy = nodes[j][1] - nodes[i][1]
                                dist = np.sqrt(dx**2 + dy**2)
                                dists.append((dist, j, dx, dy))
                        
                        # Sort by distance
                        dists.sort(key=lambda x: x[0])
                        # Take k=2 nearest neighbors
                        k = min(2, len(dists))
                        for n_idx in range(k):
                            dist, j, dx, dy = dists[n_idx]
                            rel_w = nodes[j][2] / (nodes[i][2] + 1e-6)
                            rel_h = nodes[j][3] / (nodes[i][3] + 1e-6)
                            
                            edges.append([i, j])
                            edge_feats.append([dx, dy, dist, rel_w, rel_h])
                            
                edge_indices_tensor = torch.tensor(edges, dtype=torch.long).t() if len(edges) > 0 else torch.zeros((2, 0), dtype=torch.long)
                edge_features_tensor = torch.tensor(edge_feats, dtype=torch.float32) if len(edge_feats) > 0 else torch.zeros((0, 5), dtype=torch.float32)
                
                graph_dict = {
                    'nodes': nodes_tensor,
                    'edge_indices': edge_indices_tensor,
                    'edge_features': edge_features_tensor
                }
                return image, target_vec, morphology_vec, graph_dict
                
            return image, target_vec, morphology_vec
            
        return image, target_vec
