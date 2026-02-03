
import torch
import numpy as np
import os
import json

from torch.utils.data import Dataset


class DrywallQADatasetCustom(Dataset):

    def __init__(self, folder_dir):
        self.json_dir = os.path.join(folder_dir, "_annotations.coco.json")
        with open(self.json_dir, 'r') as f:
            
            self.data = json.load(f)
            print("Data loaded from JSON")

    def __len__(self):
        return len(self.data['images'])

    def __getitem__(self, idx):
        self.image_id = self.data['images'][idx]['id']
        self.image_file_name = self.data['images'][idx]['file_name']

        self.image_path = os.path.join(self.json_dir, self.image_file_name)
        print(self.image_path)
        # self.image_file_name = os.path.join(self.json_dir,self.image)
        # print(self.image.keys()) 

        return {}

    def data_info(self):
        print(self.data.keys())
        print(self.data['info'])
        print("Number of Images", len(self.data['images']))
        print("Number of annotations", len(self.data['annotations']))




def main():
    d = DrywallQADatasetCustom("/home/saitama/Documents/Prompted Segmentation for Drywall QA/datasets/Drywall_Join_Detect_v1/train/")
    print(len(d))
    print(d[0])    
    # d.data_info()


if __name__ == "__main__":
    main()