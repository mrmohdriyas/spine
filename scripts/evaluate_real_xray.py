import os
import torch
import json
import pandas as pd
from PIL import Image
import torchvision
from torchvision.ops import box_iou
from models.dann_faster_rcnn import DANN_FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_source_only_model():
    num_classes = 6
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(r"c:\projects\spine\outputs\phase11_domain_adaptation\detector_v2.pt", weights_only=True))
    model.eval()
    return model

def get_dann_model():
    model = DANN_FasterRCNN(num_classes=6)
    model.faster_rcnn.load_state_dict(torch.load(r"c:\projects\spine\outputs\phase12_domain_adaptation\detector_dann.pt", weights_only=True))
    model.eval()
    return model

def compute_metrics(boxes_pred, labels_pred, scores_pred, boxes_gt, labels_gt, iou_threshold=0.5):
    # Map L1-L5 to 1-5 (or match with whatever your model outputs).
    # Assuming model outputs 1 for L1, 2 for L2, etc.
    label_map = {"L1": 1, "L2": 2, "L3": 3, "L4": 4, "L5": 5}
    
    tp_per_class = {k: 0 for k in label_map.values()}
    fp_per_class = {k: 0 for k in label_map.values()}
    fn_per_class = {k: 0 for k in label_map.values()}
    iou_sums = {k: [] for k in label_map.values()}
    
    # Track which GT boxes have been matched
    matched_gt = set()
    
    for i in range(len(boxes_pred)):
        if scores_pred[i] < 0.5:
            continue
            
        p_box = boxes_pred[i].unsqueeze(0)
        p_label = labels_pred[i].item()
        
        if p_label not in label_map.values():
            continue
            
        best_iou = 0
        best_gt_idx = -1
        
        for j in range(len(boxes_gt)):
            g_label = label_map[labels_gt[j]]
            if p_label == g_label:
                iou = box_iou(p_box, boxes_gt[j].unsqueeze(0)).item()
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
                    
        if best_iou >= iou_threshold and best_gt_idx not in matched_gt:
            tp_per_class[p_label] += 1
            matched_gt.add(best_gt_idx)
            iou_sums[p_label].append(best_iou)
        else:
            fp_per_class[p_label] += 1
            
    for j in range(len(boxes_gt)):
        if j not in matched_gt:
            g_label = label_map[labels_gt[j]]
            fn_per_class[g_label] += 1
            
    return tp_per_class, fp_per_class, fn_per_class, iou_sums

def evaluate_models(device, test_json_path, results_csv, ft_model=None):
    with open(test_json_path, 'r') as f:
        test_data = json.load(f)
        
    source_model = get_source_only_model().to(device)
    dann_model = get_dann_model().to(device)
    
    models = {
        "Source-Only": source_model,
        "DANN": dann_model.faster_rcnn
    }
    if ft_model is not None:
        models["DANN-Finetuned"] = ft_model
        
    all_results = []
    
    for model_name, model in models.items():
        total_tp = {1:0, 2:0, 3:0, 4:0, 5:0}
        total_fp = {1:0, 2:0, 3:0, 4:0, 5:0}
        total_fn = {1:0, 2:0, 3:0, 4:0, 5:0}
        total_iou = {1:[], 2:[], 3:[], 4:[], 5:[]}
        
        with torch.no_grad():
            for item in test_data:
                img_path = os.path.join(r"c:\projects\spine\data\vindr_vertebral_annotations\images", f"{item['image_id']}.png")
                if not os.path.exists(img_path): continue
                
                img = Image.open(img_path).convert("RGB")
                img_tensor = torchvision.transforms.functional.to_tensor(img).unsqueeze(0).to(device)
                
                gt_boxes = torch.tensor([a['bbox'] for a in item['annotations']], dtype=torch.float32).to(device)
                gt_labels = [a['vertebra'] for a in item['annotations']]
                
                if len(gt_boxes) == 0: continue
                
                pred = model(img_tensor)[0]
                tp, fp, fn, iou = compute_metrics(pred['boxes'], pred['labels'], pred['scores'], gt_boxes, gt_labels)
                
                for k in total_tp.keys():
                    total_tp[k] += tp[k]
                    total_fp[k] += fp[k]
                    total_fn[k] += fn[k]
                    total_iou[k].extend(iou[k])
                    
        for k in total_tp.keys():
            precision = total_tp[k] / (total_tp[k] + total_fp[k] + 1e-8)
            recall = total_tp[k] / (total_tp[k] + total_fn[k] + 1e-8)
            mean_iou = sum(total_iou[k]) / len(total_iou[k]) if len(total_iou[k]) > 0 else 0
            
            all_results.append({
                "Model": model_name,
                "Vertebra": f"L{k}",
                "Precision": precision,
                "Recall": recall,
                "IoU": mean_iou,
                "TP": total_tp[k],
                "FP": total_fp[k],
                "FN": total_fn[k]
            })
            
    df = pd.DataFrame(all_results)
    df.to_csv(results_csv, index=False)
    print(df)
    return df

if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    out_dir = r"c:\projects\spine\outputs\phase13_real_xray_validation"
    os.makedirs(out_dir, exist_ok=True)
    evaluate_models(device, r"c:\projects\spine\data\vindr_vertebral_annotations\test.json", os.path.join(out_dir, "source_only_vs_dann.csv"))
