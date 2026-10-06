
# Dual-Graph NeuroMorphNet Architecture

```text
                        X-ray
                          ↓
                    Image Encoder
                          ↓
                 Vertebral Localization
                          ↓
                Vertebral Morphology Encoder (VME)
                          ↓
     ┌────────────────────┴────────────────────┐
     ↓                                         ↓
Anatomical Relation Graph (ARG)       Pathology Relation Graph
     │                                         │
     └────────────────────┬────────────────────┘
                          ↓
                 Dual-Graph Fusion
                          ↓
        Morphology-Aware Cross-Attention (MCAM)
                          ↓
              Multi-Pathology Prediction
                          ↓
        Uncertainty-Guided Decision Module (UGDM)
                          ↓
              Prediction + Uncertainty
                          ↓
                   Explainability
```
