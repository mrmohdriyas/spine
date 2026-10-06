# Phase 6 — Spatial Graph Analysis

## Graph Construction Statistics
The Spatial Relation Graph (SRG) connects pathology-region bounding boxes based on the 2 nearest spatial neighbors (using Euclidean distance of box centers). 

Because the dataset relies strictly on existing pathology annotations to create nodes, a large portion of the dataset lacks any graph structure.

| Metric | Value |
| --- | ---: |
| **Total Images (Train)** | 2,633 |
| **Average Nodes per Image** | 1.84 |
| **Minimum Nodes** | 0 |
| **Maximum Nodes** | 19 |
| **Average Edges per Image** | 3.29 |
| **Zero-Node Graphs** | 1,350 (51.2%) |
| **Single-Node Graphs** | 253 (9.6%) |
| **Multi-Node Graphs** | 1,030 (39.1%) |
| **Graph Density** | 0.698 (for multi-node graphs) |

## Interpretation
Over half of the dataset (1,350 images, exactly corresponding to the "No finding" class) contains zero nodes, resulting in empty graphs. Furthermore, nearly 10% of the dataset contains only a single node, meaning no spatial edges can be formed. 

As a result, the Spatial Graph Encoder in NeuroMorphNet V2 only performed meaningful message passing on 39% of the training data. For the remaining 61% of the data, the graph branch either returned a zero-vector or a projection of a single isolated node, severely limiting the network's ability to learn robust spatial relations across the dataset.
