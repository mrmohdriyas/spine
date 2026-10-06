# FINAL NEUROMORPHNET VISUAL ANALYSIS

## 1. Purpose
This document provides a comprehensive visual analysis of the Final Anatomy-Constrained Dual-Graph NeuroMorphNet architecture's performance on the VinDr-SpineXR test set. The visualizations aim to diagnose the behavior of the multi-label classification head, focusing particularly on the impacts of class imbalance and the lack of reliable anatomical supervision during the final validation.

## 2. Dataset
The evaluation was conducted on a strictly isolated test subset from VinDr-SpineXR. The [test_class_distribution.png](../outputs/final_neuromorphnet/visualizations/test_class_distribution.png) visualization clearly highlights the extreme class imbalance inherent in the medical dataset, with classes like "Osteophytes" dominating the positive examples, while others (e.g., "Surgical implant" or "Vertebral collapse") have significantly fewer positive occurrences.

## 3. Multi-label Evaluation
Because the model performs multi-label pathology classification (8 independent pathologies), traditional multi-class metrics (like accuracy) are inappropriate. We analyzed each pathology independently using F1 scores, Precision-Recall Curves, and ROC Curves to capture the model's true performance.

## 4. Overall Performance
The Final NeuroMorphNet achieved:
- **Micro F1:** 0.3918
- **Macro F1:** 0.0766
- **Weighted F1:** 0.2627

This is visually summarized in the [final_results_dashboard.png](../outputs/final_neuromorphnet/visualizations/final_results_dashboard.png).

## 5. Confusion Matrices
The [multilabel_confusion_matrix.png](../outputs/final_neuromorphnet/visualizations/multilabel_confusion_matrix.png) (and the individual matrices) demonstrate that the final model is suffering from a massive **False Negative** bias. For the majority of pathologies, the model predicts the negative class (0) almost exclusively, reflecting an inability to confidently extract positive pathology features from the input regions.

## 6. Per-class Performance
The [per_class_f1_comparison.png](../outputs/final_neuromorphnet/visualizations/per_class_f1_comparison.png) and [final_per_class_metrics.png](../outputs/final_neuromorphnet/visualizations/final_per_class_metrics.png) show the breakdown across the 8 pathologies. Classes with slightly higher prevalence (e.g., Osteophytes) achieved minimal F1 scores, but precision and recall completely collapsed (F1 = 0) for the majority of the less frequent classes.

## 7. ROC Analysis
The [roc_curves.png](../outputs/final_neuromorphnet/visualizations/roc_curves.png) plots the True Positive Rate against the False Positive Rate. Several classes registered an AUC near 0.5 (random guessing) or returned `N/A` due to the lack of successful positive discriminators in the evaluated test set, confirming the collapse of the feature extractor.

## 8. Precision-Recall Analysis
The [precision_recall_curves.png](../outputs/final_neuromorphnet/visualizations/precision_recall_curves.png) provides an even harsher assessment. Given the dataset imbalance, the Average Precision (AP) for most classes is extremely poor, tracking close to the baseline prevalence. This indicates the model cannot maintain precision when recall increases.

## 9. Error Analysis
Top error cases are extracted into [error_cases/top_errors_summary.csv](../outputs/final_neuromorphnet/visualizations/error_cases/top_errors_summary.csv). The [prediction_probability_distribution.png](../outputs/final_neuromorphnet/visualizations/prediction_probability_distribution.png) violin plots reveal why the false negative rate is so high: the predicted probabilities for almost all test samples cluster heavily near 0.0, well below the 0.5 decision threshold, regardless of the true label.

## 10. Uncertainty Analysis
The [uncertainty_by_class.png](../outputs/final_neuromorphnet/visualizations/uncertainty_by_class.png) chart shows the mean uncertainty (measured via Monte Carlo Dropout in the UGDM) for each pathology. The uncertainty remains high relative to the predicted probabilities, reflecting the network's lack of confidence in the extracted morphology.

## 11. Grad-CAM
**NOT AVAILABLE.** Due to the complexity of backpropagating gradients through the GNN-based dual-graph message passing mechanism (Anatomical Relation Graph + Pathology Relation Graph), standard Grad-CAM hooks were not fully compatible with the final unified architecture for rapid visual extraction without altering the model codebase.

## 12. Model Comparison
As shown in [model_comparison_f1.png](../outputs/final_neuromorphnet/visualizations/model_comparison_f1.png), the Final NeuroMorphNet performed worse than both the ResNet18 baseline and the clean NeuroMorphNet V3 prototype.
- **ResNet18 Micro F1:** 0.4000
- **NeuroMorphNet V3 Micro F1:** 0.4906
- **Final NeuroMorphNet Micro F1:** 0.3918

## 13. Anatomical Limitation
The failure of the final model is directly attributable to the lack of validated anatomical localization. Because the vertebral localizer could not accurately extract L1-L5 bounding boxes on real X-rays without ground-truth human annotations (and was frozen to prevent environment timeouts), the VME extracted morphology from arbitrary spatial regions.

## 14. Interpretation
The final architecture was evaluated under severely limited anatomical supervision. The dual graph has **not** been proven superior for pathology classification in this experiment, because the foundational anatomical nodes passed to the graph were essentially random noise. 

The project successfully engineered a complex, leakage-free Dual-Graph pipeline (as seen in [architecture_diagram.png](../outputs/final_neuromorphnet/visualizations/architecture_diagram.png)), but experimental validation of its superiority requires an accurate upstream vertebral localizer or manual anatomical annotations on the VinDr-SpineXR dataset.
