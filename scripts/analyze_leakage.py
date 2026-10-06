import os
import json
import pandas as pd
import numpy as np
import sys
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, r"c:\projects\spine")
from src.dataset.spine_dataset import SpineDataset

def analyze():
    print("Loading dataset...")
    # Using training set to calculate statistics
    ds = SpineDataset(r"c:\projects\spine\data\manifests\train.csv", 
                      r"c:\projects\spine\data\class_mapping.json", 
                      transforms=None, return_morphology=True, return_graph=True)
                      
    with open(r"c:\projects\spine\data\class_mapping.json", 'r') as f:
        class_mapping = json.load(f)
        
    class_names = [v for k, v in sorted(class_mapping.items(), key=lambda item: int(item[0]))]
    
    # Store data
    all_targets = []
    all_morphs = []
    
    # Graph stats
    node_counts = []
    edge_counts = []
    
    print("Computing features for all training images...")
    for i in range(len(ds.image_ids)):
        img_id = ds.image_ids[i]
        group = ds.image_groups.get_group(img_id)
        
        # We don't want to load images just to get graph sizes, so we manually do it
        orig_w, orig_h = 512, 512 # approx for fast stats
        # but to be accurate we can just use the target generation logic
        
        target_vec = np.zeros(ds.num_classes, dtype=np.float32)
        has_pathology = False
        for _, row in group.iterrows():
            lbl_id = int(row['label_id'])
            target_vec[lbl_id] = 1.0
            if lbl_id != ds.no_finding_idx:
                has_pathology = True
                
        if ds.no_finding_idx != -1:
            if has_pathology:
                target_vec[ds.no_finding_idx] = 0.0
            elif target_vec.sum() == 0:
                target_vec[ds.no_finding_idx] = 1.0
                
        all_targets.append(target_vec)
        
        morphology_vec = np.zeros(8, dtype=np.float32)
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
            
        all_morphs.append(morphology_vec)
        
        # Graph
        N = len(nodes)
        node_counts.append(N)
        
        if N > 1:
            edges = []
            for j in range(N):
                dists = []
                for k_idx in range(N):
                    if j != k_idx:
                        dx = nodes[k_idx][0] - nodes[j][0]
                        dy = nodes[k_idx][1] - nodes[j][1]
                        dist = np.sqrt(dx**2 + dy**2)
                        dists.append((dist, k_idx))
                dists.sort(key=lambda x: x[0])
                k_neighbors = min(2, len(dists))
                for n_idx in range(k_neighbors):
                    edges.append([j, dists[n_idx][1]])
            edge_counts.append(len(edges))
        else:
            edge_counts.append(0)
            
    all_targets = np.array(all_targets)
    all_morphs = np.array(all_morphs)
    
    # Feature correlations
    print("\n--- Leakage Analysis (Pearson Correlation) ---")
    feature_names = ['mean_w', 'mean_h', 'mean_area', 'mean_ar', 'mean_cx', 'mean_cy', 'max_area', 'num_annotations']
    
    leakage_results = {}
    
    for i, cname in enumerate(class_names):
        print(f"\nClass: {cname}")
        y = all_targets[:, i]
        
        # If class is too rare, skip
        if y.sum() < 5:
            continue
            
        class_leakage = {}
        for j, fname in enumerate(feature_names):
            x = all_morphs[:, j]
            # Point-biserial correlation
            corr = np.corrcoef(x, y)[0, 1]
            if np.isnan(corr): corr = 0.0
            
            # Mean for positive vs negative
            pos_mean = x[y == 1].mean()
            neg_mean = x[y == 0].mean()
            
            class_leakage[fname] = {
                "correlation": float(corr),
                "pos_mean": float(pos_mean),
                "neg_mean": float(neg_mean)
            }
            if abs(corr) > 0.4:
                print(f"  WARNING: {fname} correlates strongly ({corr:.2f}). Pos mean: {pos_mean:.2f}, Neg mean: {neg_mean:.2f}")
        leakage_results[cname] = class_leakage
        
    with open(r"c:\projects\spine\docs\leakage_analysis.json", 'w') as f:
        json.dump(leakage_results, f, indent=4)
        
    # Graph Statistics
    node_counts = np.array(node_counts)
    edge_counts = np.array(edge_counts)
    
    total_graphs = len(node_counts)
    single_node = np.sum(node_counts == 1)
    multi_node = np.sum(node_counts > 1)
    zero_node = np.sum(node_counts == 0)
    
    graph_stats = {
        "average_nodes": float(np.mean(node_counts)),
        "min_nodes": int(np.min(node_counts)),
        "max_nodes": int(np.max(node_counts)),
        "average_edges": float(np.mean(edge_counts)),
        "single_node_graphs": int(single_node),
        "multi_node_graphs": int(multi_node),
        "zero_node_graphs": int(zero_node),
        "graph_density": float(np.mean(edge_counts[node_counts > 1] / (node_counts[node_counts > 1] * (node_counts[node_counts > 1] - 1)))) if multi_node > 0 else 0.0
    }
    
    with open(r"c:\projects\spine\docs\graph_statistics.json", 'w') as f:
        json.dump(graph_stats, f, indent=4)
        
    print("\n--- Graph Statistics ---")
    for k, v in graph_stats.items():
        print(f"{k}: {v}")
        
if __name__ == '__main__':
    analyze()
