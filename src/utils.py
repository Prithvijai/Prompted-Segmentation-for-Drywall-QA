import torch
import numpy as np
import torchvision.transforms.functional as TF

def collate_fn(batch):
    """for pre-processing the dataset before loading"""
    size = (640, 640)
    images = [TF.resize(item["image"], size) for item in batch]
    images = [TF.to_tensor(img) for img in images]

    masks = [TF.resize(item["mask"], size, interpolation=TF.InterpolationMode.NEAREST) for item in batch]
    masks = [torch.tensor(np.array(m), dtype=torch.float32) / 255.0 for m in masks]

    prompts = [item["prompt"] for item in batch]
    
    return {
        "image": images,
        "mask": torch.stack(masks), 
        "prompt": prompts
    }


def calculate_metrics(pred_mask, gt_mask):
    """Calculates IoU and Dice Score for binary masks"""
    pred = (pred_mask > 0.5).astype(np.uint8)
    gt = (gt_mask > 0.5).astype(np.uint8)
    
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    
    iou = intersection / union if union > 0 else 1.0
    dice = (2 * intersection) / (pred.sum() + gt.sum()) if (pred.sum() + gt.sum()) > 0 else 1.0
    
    return iou, dice