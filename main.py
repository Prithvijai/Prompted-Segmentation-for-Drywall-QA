
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader
from src.preprocessing import DrywallQADatasetCustom
import matplotlib.pyplot as plt


def main():
    d1_train = DrywallQADatasetCustom("/home/saitama/Documents/Prompted Segmentation for Drywall QA/datasets/Drywall_Join_Detect_v1/train/")
    d2_train = DrywallQADatasetCustom("/home/saitama/Documents/Prompted Segmentation for Drywall QA/datasets/cracks_v1/train")
    # print(len(d))   
    image1, mask1 = d1_train[0]['image'], d1_train[0]['mask']
    image2, mask2 = d2_train[0]['image'], d2_train[0]['mask']
    # print(annotations)
    fig, ax = plt.subplots(2, 2, figsize=(12, 12))
    ax = ax.flatten()
    ax[0].imshow(image1)
    ax[1].imshow(image2)
    ax[2].imshow(mask1)
    ax[3].imshow(mask2)
    plt.show()
    # d.data_info()


if __name__ == "__main__":
    main()