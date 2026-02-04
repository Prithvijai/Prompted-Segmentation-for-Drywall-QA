import torch
import numpy as np
import torchvision.transforms.functional as TF

def collate_fn(batch):
    size = (640, 640)
    # for faster trainig half the 640 * 640 resolution
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
