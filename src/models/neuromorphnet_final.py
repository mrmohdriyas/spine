import torch
import torch.nn as nn
import torchvision

from .vertebral_localizer import VertebralLocalizer
from .vertebral_morphology_encoder import VertebralMorphologyEncoder
from .anatomical_relation_graph import AnatomicalRelationGraph
from .pathology_relation_graph import PathologyRelationGraph
from .dual_graph_fusion import DualGraphFusion
from .mcam import MorphologyAwareCrossAttention
from .pathology_head import MultiPathologyHead
from .uncertainty_module import UncertaintyGuidedDecisionModule

class AnatomyConstrainedDualGraphNeuroMorphNet(nn.Module):
    """
    Part L: The Final Complete Architecture.
    End-to-End model encompassing all Phase 14 requirements.
    """
    def __init__(self, embed_dim=256, num_pathologies=8):
        super().__init__()
        
        # 1. Global Image Backbone (e.g., ResNet50 up to layer3 for spatial features)
        resnet = torchvision.models.resnet50(weights=None)
        self.backbone = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3 # 1024 channels output
        )
        self.backbone_proj = nn.Conv2d(1024, embed_dim, kernel_size=1)
        
        # 2. Vertebral Localizer
        self.localizer = VertebralLocalizer(num_classes=6)
        
        # 3. Vertebral Morphology Encoder (VME)
        self.vme = VertebralMorphologyEncoder(feature_dim=embed_dim)
        
        # 4. Anatomical Relation Graph (ARG)
        self.arg = AnatomicalRelationGraph(node_dim=embed_dim, edge_dim=4, hidden_dim=256)
        
        # 5. Pathology Relation Graph
        # Backbone outputs 1024 channels. 
        self.pathology_graph = PathologyRelationGraph(in_channels=1024, node_dim=embed_dim, num_pathologies=num_pathologies)
        
        # 6. Dual-Graph Fusion
        self.fusion = DualGraphFusion(embed_dim=embed_dim)
        
        # 7. MCAM
        self.mcam = MorphologyAwareCrossAttention(embed_dim=embed_dim)
        
        # 8. Multi-Pathology Head
        self.head = MultiPathologyHead(embed_dim=embed_dim, num_classes=num_pathologies)
        
        # 9. UGDM
        self.ugdm = UncertaintyGuidedDecisionModule(num_mc_passes=10)
        
    def forward(self, images, targets=None, return_uncertainty=False):
        """
        Args:
            images: Tensor [B, C, H, W]
            targets: dict containing ground truth (only used for training losses if applicable)
            return_uncertainty: bool, if True uses MC Dropout to estimate variance
            
        Returns:
            dict containing logits, probs, graphs, and optionally uncertainty.
        """
        B = images.size(0)
        
        # --- PATH A: Anatomical Branch ---
        # A1. Localize Vertebrae
        # In an ideal end-to-end setting, we'd backprop through boxes.
        # But ROI Align expects coordinates. We use the localizer in inference mode 
        # to generate regions for the VME, stopping gradients to the localizer box coords.
        localizer_out = self.localizer.extract_regions(images, conf_threshold=0.5)
        boxes_list = [out['boxes'] for out in localizer_out]
        labels_list = [out['labels'] for out in localizer_out]
        
        # A2. Vertebral Morphology Encoder (VME)
        vme_features_list = self.vme(images, boxes_list)
        
        # A3. Anatomical Relation Graph (ARG)
        arg_nodes_list = []
        global_spine_list = []
        for i in range(B):
            refined_nodes, global_spine = self.arg(vme_features_list[i], boxes_list[i], labels_list[i])
            arg_nodes_list.append(refined_nodes)
            global_spine_list.append(global_spine)
            
        # --- PATH B: Pathology Branch ---
        # B1. Backbone Features
        global_features = self.backbone(images) # [B, 1024, H/16, W/16]
        proj_features = self.backbone_proj(global_features) # [B, embed_dim, H/16, W/16]
        
        # B2. Pathology Relation Graph
        path_graph_nodes, region_logits = self.pathology_graph(global_features) # [B, G*G, embed_dim]
        
        # --- PATH C: Fusion & Final Decision ---
        # C1. Dual-Graph Fusion
        fused_path_graph = self.fusion(arg_nodes_list, path_graph_nodes) # [B, G*G, embed_dim]
        
        # C2. MCAM
        final_embedding = self.mcam(fused_path_graph, proj_features, vme_features_list) # [B, embed_dim]
        
        # C3. Prediction
        if return_uncertainty:
            # Uses MC Dropout over the head
            ugdm_out = self.ugdm(self.head, final_embedding)
            return ugdm_out
        else:
            logits, loss = self.head(final_embedding, targets)
            return {
                "logits": logits,
                "probs": torch.sigmoid(logits),
                "loss": loss
            }
