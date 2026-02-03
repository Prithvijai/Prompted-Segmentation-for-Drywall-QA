
import torch
import numpy as np
import os
import json

from torch.utils.data import Dataset
from pycocotools.coco import COCO
from PIL import Image

class DrywallQADatasetCustom(Dataset):
    """ Loads COCO format data
    
    load the images and annotations from COCO format (data folder). 
    convert the bbox from annotations into masks if there is no segementations
     available for the image. 
    """
    def __init__(self, folder_dir, prompt):
        self.prompt = prompt
        self.folder_dir = folder_dir
        self.json_dir = os.path.join(folder_dir, "_annotations.coco.json")
        self.coco = COCO(self.json_dir)

        with open(self.json_dir, 'r') as f:
            
            self.data = json.load(f)
            print("Data loaded from JSON")

    def __len__(self):
        return len(self.data['images'])

    def __getitem__(self, idx):
        image_id = self.data['images'][idx]['id']
        image_file_name = self.data['images'][idx]['file_name']

        image_path = os.path.join(self.folder_dir, image_file_name)
        annotation_ids = self.coco.getAnnIds(image_id)
        annotations = self.coco.loadAnns(annotation_ids)

        image = Image.open(image_path).convert("RGB")

        w , h = image.size
        mask = np.zeros((w, h), dtype=np.uint8)

        for a in annotations:
            if 'segmentation' in a and a['segmentation']:
                mask = np.maximum(mask, self.coco.annToMask(a))
            elif 'bbox' in a:
                x, y, w_box, h_box = [int(v) for v in a['bbox']]
                mask[y : y + h_box, x: x + w_box] = 1

        mask = (mask * 255).astype(np.uint8)
        return { "image": image, "mask": Image.fromarray(mask), "prompt" : self.prompt, "image_id" : image_id}

    def data_info(self):
        print(self.data.keys())
        print(self.data['info'])
        print("Number of Images", len(self.data['images']))
        print("Number of annotations", len(self.data['annotations']))




