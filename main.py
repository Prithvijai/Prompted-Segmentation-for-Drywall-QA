
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader
from src.preprocessing import DrywallQADatasetCustom
import matplotlib.pyplot as plt


def plot_overlay(axis, image, mask, title):
        axis.imshow(image)
        axis.imshow(mask, alpha=0.5, cmap='jet') 
        axis.set_title(title)
        axis.axis('off')

def main():
    d1_train = DrywallQADatasetCustom("/home/saitama/Documents/Prompted Segmentation for Drywall QA/datasets/Drywall_Join_Detect_v1/train/")
    d2_train = DrywallQADatasetCustom("/home/saitama/Documents/Prompted Segmentation for Drywall QA/datasets/cracks_v1/train")
    # print(len(d))   
    image1, mask1 = d1_train[0]['image'], d1_train[0]['mask']
    image2, mask2 = d2_train[0]['image'], d2_train[0]['mask']
    # print(annotations)
    fig, ax = plt.subplots(1, 2, figsize=(16, 8)) 
    
    plot_overlay(ax[0], image1, mask1, "Taping Area Detection")
    plot_overlay(ax[1], image2, mask2, "Wall Crack Detection")

    plt.tight_layout()
    plt.show()
    # d.data_info()


if __name__ == "__main__":
    main()