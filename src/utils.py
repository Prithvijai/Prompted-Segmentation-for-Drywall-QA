import torch
import numpy as np
import torchvision.transforms.functional as TF

def collate_fn(batch):
    images = [TF.to_tensor(item["image"]) for item in batch]
    masks = [torch.tensor(np.array(item["mask"]), dtype=torch.float32) for item in batch]
    prompts = [item["prompt"] for item in batch]
    
    return {
        "image": images,
        "mask": torch.stack(masks), 
        "prompt": prompts
    }
